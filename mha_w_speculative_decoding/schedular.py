import torch
import torch.nn as nn


class ActiveRequest:
    def __init__(self, context_length, num_layers, prompt_token_id, max_new_tokens, request_id=None):
        self.request_id = request_id
        self.prompt_token_id = prompt_token_id
        self.pos = 0 # useful for both RoPE cslculation and for 
        self.status = "queued" # other options: "prefill", "decoding", "completed"
        self.generated_token_id = None
        self.prefill_tok_left = prompt_token_id.shape[1]
        self.max_new_tokens = max_new_tokens
        self.tokens_generated = 0


class Schedular:
    def __init__(self, max_num_batched_tokens):
        self.active_requests = {} # request_id ---> requests
        self.max_num_batched_tokens = max_num_batched_tokens

    def queue(self, context_length, num_layers, prompt_token_id, max_new_tokens, request_id):
        request = ActiveRequest(context_length, num_layers, prompt_token_id, max_new_tokens, request_id)
        self.active_requests[request_id] = request # new dictionry entry

    def next_seq(self):
        inference_seq = None # will contain the next sequence
        partition = {} # for decoding there is going to be everything in kvcache, for prefil phase there is going to be everything inclusing most recent ones (mask helps here)
        num_tok = 0


        # gather all decoding sequences
        for request_id, request in self.active_requests.items():
            if request.status == "decoding":
                next_id = request.generated_token_id[:,-1:] # in decoding phase, we only take the last token that was generated
                if inference_seq is None:
                    inference_seq = next_id
                else:
                    inference_seq = torch.cat((inference_seq, next_id), dim=-1)
                partition[(num_tok, num_tok+1)] = request_id # should use request_id as the identifier, helpful in othe rplaces
                num_tok += 1
            # print(f"request_id: {request.request_id}, cache_len: {page_allocator.cache_len(request.request_id)}, page_table keys: {[k for k in page_allocator.page_table.keys() if k[1]==0]}")
                

        # now add prefill ones
        for request_id, request in self.active_requests.items():
            if (request.status == "prefill") and (num_tok < self.max_num_batched_tokens):

                # get the tokens from the rqeuest that can be part of the sequence
                start_pos = -request.prefill_tok_left
                end_pos = min(0, -request.prefill_tok_left + (self.max_num_batched_tokens - num_tok))
                if end_pos == 0: # handling the special case, needed for indexing tensor as we are counting from end
                    end_pos = None
                if inference_seq is None:
                    inference_seq = request.prompt_token_id[:, start_pos:end_pos]
                else:
                    inference_seq = torch.cat([inference_seq, request.prompt_token_id[:, start_pos:end_pos]], dim=-1)

                # accounting
                tokens_to_add = min(request.prefill_tok_left, self.max_num_batched_tokens - num_tok)
                partition[(num_tok, num_tok + tokens_to_add)] = request_id # inclusive of first coordinate only.
                request.prefill_tok_left -= tokens_to_add
                num_tok += tokens_to_add


        # move a queued ones to prefill if space still remaining in the sequence to be processed
        for request_id, request in self.active_requests.items():
            if (request.status == "queued") and (num_tok < self.max_num_batched_tokens):
                request.status = "prefill" # update the status

                # get the tokens from the rqeuest that can be part of the sequence
                start_pos = -request.prefill_tok_left
                end_pos = min(0, -request.prefill_tok_left + (self.max_num_batched_tokens - num_tok))
                if end_pos == 0: # handling the special case, needed for indexing tensor as we are counting from end
                    end_pos = None
                if inference_seq is None:
                    inference_seq = request.prompt_token_id[:, start_pos:end_pos]
                else:
                    inference_seq = torch.cat([inference_seq, request.prompt_token_id[:, start_pos:end_pos]], dim=-1)
                
                # accounting 
                tokens_to_add = min(request.prefill_tok_left, self.max_num_batched_tokens - num_tok)
                partition[(num_tok, num_tok + tokens_to_add)] = request_id
                request.prefill_tok_left -= tokens_to_add
                num_tok += tokens_to_add

        # print(f"num_col: {num_col}, num_tok: {num_tok}")

        print(f"Batch: {num_tok} tokens | Decode: {sum(1 for request in self.active_requests.values() if request.status == 'decoding')} | Prefill: {sum(1 for request in self.active_requests.values() if request.status == 'prefill')} | Partition: {partition}")

        return inference_seq, partition, self.active_requests

    
    # indices have to be set here, not list
    def clear_completed(self, request_ids):

        # send an alert
        for request_id in request_ids:
            if self.active_requests[request_id].status != "completed":
                assert False, "Request is still active, wrong request ids, check again!"
        
        # actual cleaning
        for request_id in request_ids:
            del self.active_requests[request_id]

        



