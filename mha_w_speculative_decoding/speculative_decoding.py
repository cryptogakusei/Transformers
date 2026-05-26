import torch
import torch.nn as nn


# DraftRunner ---> TargetRunner ---> Sampler ---> TargetRunner ---> Server
#                                                       |
#                                                       |
#                                                       \/
#                                                  DraftRunner

# following this paper: https://arxiv.org/pdf/2211.17192


class SpeculativeDecodingManager:
# we use it for ensuring he draft and target systems are in synce after each run
    def __init__(self, draft_model, draft_page_allocator, target_model, target_page_allocator, num_speculations):
        self.draft_model = draft_model
        self.target_model= target_model
        self.draft_page_allocator = draft_page_allocator
        self.target_page_allocator = target_page_allocator
        self.num_speculations = num_speculations
    
    def run(self, inference_seq, partition, active_requests, num_layers):

        # savind positins from requests so as to update them later properly based on rejection
        saved_pos = {}
        for (_, _), request_id in partition.items():
            saved_pos[request_id] = active_requests[request_id].pos

        # get the intermediate sequences and the storage location used for KV caches for draft model associated with the speculative tokens
        speculated_tokens, mem_assignment_draft_model = self.run_draft_model(inference_seq, partition, active_requests)

        # construct the intermediate sequences
        verification_seq, verification_partition = self.construct_verification_seq(inference_seq, partition, speculated_tokens)

        # get the accepted tokens and the storage location used for KV caches for target model associated with the speculative tokens 
        temp_new_kvcache_target_model, target_probabilities = self.run_target_model(verification_seq, verification_partition, active_requests)

        # speculative sampling
        self.speculative_sampling(speculated_tokens, target_probabilities)

        # cleanup
        self.clean_mem_allocations(speculated_tokens, mem_assignment_draft_model, temp_new_kvcache_target_model, num_layers, active_requests, partition, saved_pos)

        # update the generated tokens field for the requests
        self.fill_requests_with_generated_tokens(speculated_tokens, active_requests)
    
    
    # returns the intermediate sequence for each request that is being fed, and 
    # the memory location of the KV caches of the speculated tokens in the draft model.
    def run_draft_model(self, inference_seq, partition, active_requests):
        print(f"num_speculations={self.num_speculations}")

        speculated_tokens = {} # request_id --> [[probas1, token1, status], ...], 
        mem_assignment_draft_model = {} # (request_id, layer) --> [[(page, cell_number), .....]_first_seq, [(page, cell_number), .....]_second_seq, ..., [(page, cell_number), .....]_num_speculations], each elment in this list identifies with a speculation until num_speculations   
        

        # loop over to compute speculations
        for _ in range(self.num_speculations):

            # logit calculation
            with torch.no_grad():
                logits = self.draft_model(
                    inference_seq, 
                    partition, 
                    active_requests, 
                    self.draft_page_allocator, 
                    mem_assignment_draft_model)
            
            next_partition = {}
            next_inference_seq = {}

            #  getting next token for each request in the original sequence
            request_counter = 0
            next_inference_seq = None
            for (_, end_pos), request_id in partition.items():
                # get the next token
                next_logits = logits[:, end_pos-1, :]
                draft_probas = torch.softmax(next_logits, dim=-1)
                next_token = torch.argmax(draft_probas, dim=-1, keepdim=True)

                # update the inference for next round
                next_inference_seq = next_token if next_inference_seq is None \
                                    else torch.cat((next_inference_seq, next_token), dim=-1)
                next_partition[(request_counter, request_counter+1)] = request_id

                # update the speculated tokens
                if request_id not in speculated_tokens:
                    speculated_tokens[request_id] = []
                speculated_tokens[request_id].append([draft_probas, next_token, "proposed"])

                request_counter += 1

            # update inference sequence for next loop
            partition = next_partition
            inference_seq = next_inference_seq
        
        return speculated_tokens, mem_assignment_draft_model


    def construct_verification_seq(self, original_inference_seq, original_partition, speculated_tokens):
        verification_seq = None
        verification_partition = {}
        pos_counter = 0

        for (start_pos, end_pos), request_id in original_partition.items():

            sequence_per_request = original_inference_seq[:, start_pos:end_pos]

            for index in range(self.num_speculations):
                [_, next_token, _] = speculated_tokens[request_id][index]
                sequence_per_request = torch.cat((sequence_per_request, next_token), dim=-1)

            verification_seq = sequence_per_request if verification_seq is None \
                            else torch.cat((verification_seq, sequence_per_request), dim=-1)
            
            verification_partition[(pos_counter, pos_counter + sequence_per_request.shape[1])] = request_id
            pos_counter += sequence_per_request.shape[1]

        return verification_seq, verification_partition


    # def construct_all_intermediate_seq(self, original_inference_seq, original_partition, speculated_tokens):
    #     intermediate_sequences = {} # speculation_number ----> (inference_seq, partition)
    #     intermediate_sequences[0] = (original_inference_seq, original_partition)


    #     for speculation_counter in range(1, self.num_speculations + 1):
    #         inference_seq = None 
    #         partition = {}
    #         pos_counter = 0 # this will help in creating partitions

    #         # iterate over all request_id that was included in original inference seq sent to draft model
    #         for (start_pos, end_pos), request_id in original_partition.items():
    #             inference_per_request = original_inference_seq[:,start_pos:end_pos] # prefix the original one

    #             # iterate over all speculations to add speculated tokens
    #             for index in range(speculation_counter): 
    #                 (_, next_token, _) = speculated_tokens[request_id][index]
    #                 inference_per_request = torch.cat((inference_per_request, next_token), dim=-1) # add the speculated token 
                
    #             # add it to the inference_seq
    #             inference_seq = inference_per_request if inference_seq is None else torch.cat((inference_seq, inference_per_request), dim=-1)

    #             # create the partition 
    #             partition[(pos_counter, pos_counter + (end_pos - start_pos) + speculation_counter)] = request_id

    #             # update the counter for next request_id
    #             pos_counter += inference_per_request.shape[1]
            
    #         # record it
    #         intermediate_sequences[speculation_counter] = (inference_seq, partition)
        
    #     return intermediate_sequences



    def run_target_model(self, verification_seq, verification_partition, active_requests):

        temp_new_kvcache_target_model = {} # (request_id, layer) --> [(k_cache, v_cache)_1, ...., (k_cache, v_cache)_num_speculations]
        target_probabilities = {} # request_id --> [probas1, probas2, ..., probas_num_speculations]

        # get next token as per target model
        with torch.no_grad():
            logits = self.target_model(
                verification_seq,
                verification_partition,
                active_requests,
                self.target_page_allocator,
                temp_new_kvcache_target_model
            )

        # update the status in speculated token data structure            
        for (start_pos, end_pos), request_id in verification_partition.items():
            target_probabilities[request_id] = []

            num_original = (end_pos - start_pos) - self.num_speculations
            for i in range(self.num_speculations + 1):
                pos = start_pos + num_original - 1 + i
                probas = torch.softmax(logits[:, pos, :], dim=-1)
                target_probabilities[request_id].append(probas)

        return temp_new_kvcache_target_model, target_probabilities

            
    def speculative_sampling(self, speculated_tokens, target_probabilities):
        for request_id, _ in target_probabilities.items():
            not_accepted = self.num_speculations
            for speculation_counter in reversed(range(0, self.num_speculations)):

                # do the comparison to get the minimum (x is the speculated token)
                [draft_probas, speculated_token, _] = speculated_tokens[request_id][speculation_counter]
                target_probas = target_probabilities[request_id][speculation_counter]
                
                # compute the uniform distribution 
                r = torch.rand(1, device=target_probas.device)
                
                if r > (target_probas[0, speculated_token.item()]/draft_probas[0, speculated_token.item()]):
                    not_accepted = speculation_counter

            # do the comparison of n < gamma and compute p' if needed
            corrected_probas = target_probabilities[request_id][self.num_speculations]
            if not_accepted < self.num_speculations:
                target_probas = target_probabilities[request_id][not_accepted]
                [draft_probas, _, _] = speculated_tokens[request_id][not_accepted]
                residual = torch.clamp(target_probas - draft_probas, min=0.0)
                total = residual.sum()
                corrected_probas = residual / total

            # update the speculated token
            bonus_token = torch.argmax(corrected_probas, dim=-1, keepdim=True)

            # update the status
            if not_accepted > 0:
                for index in range(not_accepted):
                    speculated_tokens[request_id][index][2] = "accepted"
            
            if not_accepted < self.num_speculations: 
                speculated_tokens[request_id][not_accepted][2] = "bonus"
                speculated_tokens[request_id][not_accepted][1] = bonus_token
            else: # for not_accepted = self.num_speculations
                speculated_tokens[request_id].append([None, bonus_token, "bonus"])
            
            if not_accepted < self.num_speculations:
                for index in range(not_accepted+1, self.num_speculations):
                    speculated_tokens[request_id][index][2] = "rejected"




    def clean_mem_allocations(self, speculated_tokens, mem_assignment_draft_model, temp_new_kvcache_target_model, num_layers, active_requests, original_partition, saved_pos):
        
        for request_id in list(speculated_tokens):
            # figure out how many were accepted
            not_accepted = self.num_speculations
            for index, [_, _, status] in enumerate(speculated_tokens[request_id]):
                if status == "bonus":
                    not_accepted = index
                    break

            # figure out original sequence length for this request
            num_original = 0
            for (start_pos, end_pos), rid in original_partition.items():
                if rid == request_id:
                    num_original = end_pos - start_pos
                    break

            num_to_commit = num_original + not_accepted

            for layer in range(num_layers):
                kv_entry = temp_new_kvcache_target_model[(request_id, layer)][0]
                keys_to_commit = kv_entry[0][:, :, :num_to_commit, :]
                values_to_commit = kv_entry[1][:, :, :num_to_commit, :]
                self.target_page_allocator.allocate(keys_to_commit, values_to_commit, layer, request_id)

            # draft model cleanup, reclaim rejected speculations
            if active_requests[request_id].status != "prefill":
                for index in range(1, len(mem_assignment_draft_model[(request_id, 0)])):
                    [_, _, status] = speculated_tokens[request_id][index - 1]
                    if status in ("bonus", "rejected"):
                        for layer in range(num_layers):
                            mem_assignment = mem_assignment_draft_model[(request_id, layer)][index]
                            self.draft_page_allocator.reclaim_cellwise(mem_assignment, request_id, layer)
            else:
                for index in range(1, len(mem_assignment_draft_model[(request_id, 0)])):
                    for layer in range(num_layers):
                        print(f"request_id={request_id}, layer={layer}, index={index}, len={len(mem_assignment_draft_model[(request_id, layer)])}, num_speculations={self.num_speculations}")
                        mem_assignment = mem_assignment_draft_model[(request_id, layer)][index]
                        self.draft_page_allocator.reclaim_cellwise(mem_assignment, request_id, layer)


            active_requests[request_id].pos = saved_pos[request_id] + num_original + not_accepted + 1
            



    # def clean_mem_allocations(self, speculated_tokens, mem_assignment_draft_model, temp_new_kvcache_target_model, num_layers, active_requests):
        
    #     # load the kv cache from the original sequence (first one is exception)
    #     for (request_id, layer) in list(temp_new_kvcache_target_model):
    #         new_keys = temp_new_kvcache_target_model[(request_id, layer)][0][0]
    #         new_values = temp_new_kvcache_target_model[(request_id, layer)][0][1]
    #         self.target_page_allocator.allocate(new_keys, new_values, layer, request_id)

    #     # now rest of it
    #     for request_id in list(speculated_tokens):
    #         if active_requests[request_id].status != "prefill":
    #             for index, [_, _, status] in enumerate(speculated_tokens[request_id][:-1]):
    #                 if (status == "accepted"):
    #                     # allocate the page allocation for target model
    #                     # no need to manage the page allocation for draft model as they are appropriately placed already
    #                     for layer in range(num_layers):
    #                         new_keys = temp_new_kvcache_target_model[(request_id, layer)][index+1][0]
    #                         new_values = temp_new_kvcache_target_model[(request_id, layer)][index+1][1]
    #                         self.target_page_allocator.allocate(new_keys, new_values, layer, request_id)


    #                 if (status == "bonus") or (status == "rejected"):
    #                     # delete the page allocations made beforehand for rejected tokens in draft model
    #                     for layer in range(num_layers):
    #                         mem_assignment = mem_assignment_draft_model[(request_id, layer)][index+1] # note that index will at max be num_speculations - 1
    #                         self.draft_page_allocator.reclaim_cellwise(mem_assignment, request_id, layer)

    #         else:
    #             # we do clean up for all speclated tokens if the request is still in prefill phase, except the first one as that contains prefill chunk
    #             for index, [_, _, status] in enumerate(speculated_tokens[request_id][:-1]):
    #                 for layer in range(num_layers):
    #                     mem_assignment = mem_assignment_draft_model[(request_id, layer)][index+1] # note that index will at max be num_speculations - 1
    #                     self.draft_page_allocator.reclaim_cellwise(mem_assignment, request_id, layer)




    def fill_requests_with_generated_tokens(self, speculated_tokens, active_requests):
        for request_id in list(speculated_tokens):
            if active_requests[request_id].status == "decoding" or active_requests[request_id].prefill_tok_left == 0:
                
                # get the tokens to be added for this particular request_id
                new_generated_token_ids = None
                for index in range(len(speculated_tokens[request_id])):
                    status = speculated_tokens[request_id][index][2]
                    token_id = speculated_tokens[request_id][index][1]
                    if (status == "accepted") or (status == "bonus"):
                        if new_generated_token_ids is None:
                            new_generated_token_ids = token_id
                        else:
                            new_generated_token_ids = torch.cat((new_generated_token_ids, token_id), dim=-1)

                # add now
                if active_requests[request_id].generated_token_id is None:
                    active_requests[request_id].generated_token_id = new_generated_token_ids
                else:
                    active_requests[request_id].generated_token_id = torch.cat((active_requests[request_id].generated_token_id, new_generated_token_ids), dim=-1)
                
                active_requests[request_id].tokens_generated += new_generated_token_ids.shape[-1]






