import torch
import torch.nn as nn

from kv_cache import KVCache


class ActiveRequest:
    def __init__(self, context_length, num_layers, prompt_token_id, max_new_tokens, request_id=None):
        self.request_id = request_id
        self.prompt_token_id = prompt_token_id
        self.kvcaches = [KVCache(context_length) for _ in range(num_layers)]
        self.pos = 0 # useful for both RoPE cslculation and for 
        self.status = "queued" # other options: "prefill", "decoding", "completed"
        self.generated_token_id = None
        self.prefill_tok_left = prompt_token_id.shape[1]
        self.max_new_tokens = max_new_tokens
        self.tokens_generated = 0

    def cache_len(self):
        return 0 if self.kvcaches[0].K_cache is None else self.kvcaches[0].K_cache.shape[2] 

    
class KVPool:
    def __init__(self, max_num_batched_tokens):
        self.active_requests = []
        self.max_num_batched_tokens = max_num_batched_tokens

    def allocate_cache(self, context_length, num_layers, prompt_token_id, max_new_tokens, request_id=None):
        request = ActiveRequest(context_length, num_layers, prompt_token_id, max_new_tokens, request_id)
        self.active_requests.append(request)
    

    def next_seq(self):
        inference_seq = None
        partition = {} # for decoding there is going to be everything in kvcache, for prefil phase there is going to be everything inclusing most recent ones (mask helps here)
        num_tok = 0
        num_col = 0
        mask_row_identifier = [] # for decoding ones, it is just single element. For prefill ones, it only indicates starting position
        mask_col_identifier = [] # need to consider all tokens for whom KV cache has been computed

        # gather all decoding sequences
        for i, request in enumerate(self.active_requests):
            if request.status == "decoding":
                next_id = request.generated_token_id[:,-1:]
                if inference_seq is None:
                    inference_seq = next_id
                else:
                    inference_seq = torch.cat((inference_seq, next_id), dim=-1)
                for col in range(num_col, num_col + request.cache_len() + 1):
                    mask_row_identifier.append(num_tok)
                    mask_col_identifier.append(col)
                
                partition[(num_tok, num_tok+1)] = i

                num_tok += 1
                num_col += request.cache_len() + 1 # applicable for KVCache class, 1 is there because we will add the new token too

                

        # now add prefill ones
        for i, request in enumerate(self.active_requests):
            if (request.status == "prefill") and (num_tok < self.max_num_batched_tokens):    
                start_pos = -request.prefill_tok_left
                end_pos = min(0, -request.prefill_tok_left + (self.max_num_batched_tokens - num_tok))
                if end_pos == 0: # handling the special case, needed for indexing tensor as we are counting from end
                    end_pos = None
                if inference_seq is None:
                    inference_seq = request.prompt_token_id[:, start_pos:end_pos]
                else:
                    inference_seq = torch.cat([inference_seq, request.prompt_token_id[:, start_pos:end_pos]], dim=-1)

                tokens_to_add = min(request.prefill_tok_left, self.max_num_batched_tokens - num_tok)

                for row in range(num_tok,  num_tok + tokens_to_add):
                    for col in range(num_col , num_col + request.cache_len() + (row - num_tok) + 1):
                        mask_row_identifier.append(row)
                        mask_col_identifier.append(col)

                partition[(num_tok, num_tok + tokens_to_add)] = i

                request.prefill_tok_left -= tokens_to_add
                num_col += request.cache_len() + tokens_to_add
                num_tok += tokens_to_add


        # move a queued ones to prefill if space is remaining
        for i, request in enumerate(self.active_requests):
            if (request.status == "queued") and (num_tok < self.max_num_batched_tokens):
                request.status = "prefill"

                start_pos = -request.prefill_tok_left
                end_pos = min(0, -request.prefill_tok_left + (self.max_num_batched_tokens - num_tok))
                if end_pos == 0: # handling the special case, needed for indexing tensor as we are counting from end
                    end_pos = None
                if inference_seq is None:
                    inference_seq = request.prompt_token_id[:, start_pos:end_pos]
                else:
                    inference_seq = torch.cat([inference_seq, request.prompt_token_id[:, start_pos:end_pos]], dim=-1)
                
                tokens_to_add = min(request.prefill_tok_left, self.max_num_batched_tokens - num_tok)

                for row in range(num_tok,  num_tok + tokens_to_add):
                    for col in range(num_col , num_col + request.cache_len() + (row - num_tok) + 1):
                        mask_row_identifier.append(row)
                        mask_col_identifier.append(col)

                partition[(num_tok, num_tok + tokens_to_add)] = i

                request.prefill_tok_left -= tokens_to_add
                num_col += request.cache_len() + tokens_to_add
                num_tok += tokens_to_add




        # create mask
        mask = torch.ones((num_tok, num_col), dtype=torch.bool)
        mask[mask_row_identifier, mask_col_identifier] = False

        print(f"Batch: {num_tok} tokens | Decode: {sum(1 for r in self.active_requests if r.status == 'decoding')} | Prefill: {sum(1 for r in self.active_requests if r.status == 'prefill')} | Partition: {partition}")

        return inference_seq, partition, mask, self.active_requests

    
    # indices have to be set here, not list
    def clear_completed(self, indices):

        for index in indices:
            if self.active_requests[index].status != "completed":
                assert False, "Request is still active, wrong indices"
        
        self.active_requests = [requests for i, requests in enumerate(self.active_requests) if i not in indices]


        



