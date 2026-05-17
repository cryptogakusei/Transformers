import torch 
import torch.nn as nn

from config import INFERENCE_CONFIG, MODEL_CONFIG

# follows from the paper https://arxiv.org/pdf/2309.06180

class PageAllocator:
    def __init__(self, num_pages, num_heads, tokens_per_page, head_dim, device):
        # emulating physical mem 
        self.tokens_per_page = tokens_per_page
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_pages = num_pages
        self.page_pool_keys = torch.empty(num_pages, num_heads, tokens_per_page, head_dim, device=device)  # note that each layer will have its own page also
        self.page_pool_values = torch.empty(num_pages, num_heads, tokens_per_page, head_dim, device=device) 
        self.page_table = {} # (request_id, layer) -> [page1, page2, ....] in the chronological order
        self.empty_pages = [page for page in range(num_pages)] # list of all empty pages
        self.page_space_left = {} # page -> number

    def allocate(self, new_keys, new_values, layer, request_id):
        # note that this is per layer, per request_id

        space_needed = new_keys.shape[2]
        space_consumed = 0

        if (request_id, layer) not in self.page_table:
            new_page = self.empty_pages.pop()
            self.page_table[(request_id, layer)] = [new_page]
            self.page_space_left[new_page] = self.tokens_per_page

        # should be while loop - because we migth need to occuoy more than 1 new page
        while (space_needed > 0):
            current_page = self.page_table[(request_id, layer)][-1]
            current_page_space_left = self.page_space_left[current_page]

            if current_page_space_left > 0:
                delta_space_consumed = min(space_needed, current_page_space_left)

                # get indices for R/W
                start_pos_page = self.tokens_per_page - current_page_space_left
                end_pos_page = start_pos_page + delta_space_consumed

                start_pos_new_kv = space_consumed
                end_pos_new_kv = space_consumed + delta_space_consumed

                # store the new KV
                self.page_pool_keys[current_page,:,start_pos_page:end_pos_page,:] = new_keys[0,:,start_pos_new_kv:end_pos_new_kv,:]
                self.page_pool_values[current_page,:,start_pos_page:end_pos_page,:] = new_values[0,:,start_pos_new_kv:end_pos_new_kv,:]

                # update local counters
                space_consumed += delta_space_consumed
                space_needed -= delta_space_consumed

            # update global counters
            if space_needed > 0:
                next_page = self.empty_pages.pop()
                self.page_table[(request_id, layer)].append(next_page)
                self.page_space_left[current_page] = 0
                self.page_space_left[next_page] = self.tokens_per_page
            elif space_needed == 0:
                self.page_space_left[current_page] -= delta_space_consumed

            
    
    def get_cache(self, layer, request_id):
        
        tot_tokens_cached = 0
        for page in self.page_table[(request_id, layer)]:
            if self.page_space_left[page] == 0:
                tot_tokens_cached += self.tokens_per_page
            else:
                tot_tokens_cached += self.tokens_per_page - self.page_space_left[page]

        if tot_tokens_cached == 0:
            assert False, "Total tokens cached can't be 0 for calling gather_cache" 

        k_cache = torch.empty(1,self.num_heads, tot_tokens_cached, self.head_dim, device=self.page_pool_keys.device)
        v_cache = torch.empty(1,self.num_heads, tot_tokens_cached, self.head_dim, device=self.page_pool_values.device)


        # construct caches
        start_pos = 0
        end_pos = 0
        for page in self.page_table[(request_id, layer)]:
            tokens_in_page = self.tokens_per_page - self.page_space_left[page]
            end_pos += tokens_in_page
            k_cache[:,:,start_pos:end_pos,:] = self.page_pool_keys[page,:,:tokens_in_page,:]
            v_cache[:,:,start_pos:end_pos,:] = self.page_pool_values[page,:,:tokens_in_page,:]
            start_pos += tokens_in_page
        return k_cache, v_cache


    def check_space_availability(self, request_ids, num_new_tokens_per_id, num_layers):
        # need sanity check if enough space left for all layers and all requests together before processong starts in inference engine
        # has to be called from conitnuous batching

        total_extra_pages_needed = 0
        
        for i, request_id in enumerate(request_ids):
            extra_pages_needed_per_request = 0
            if (request_id, 0) not in self.page_table:
                extra_pages_needed_per_request = (((num_new_tokens_per_id[i]) // self.tokens_per_page) + 1) * num_layers            
            else:
                space_left_current_page = self.page_space_left[self.page_table[(request_id,0)][-1]]
                if  num_new_tokens_per_id[i] > space_left_current_page:
                    extra_pages_needed_per_request = (((num_new_tokens_per_id[i] - space_left_current_page)//self.tokens_per_page) + 1) * num_layers
            total_extra_pages_needed += extra_pages_needed_per_request

        if total_extra_pages_needed <= len(self.empty_pages):
            return True
        else:
            return False


    def cache_len(self, request_id):
        total_tokens = 0
        if (request_id,0) in self.page_table:
            pages = self.page_table[(request_id,0)]
            if len(pages) > 1:
                total_tokens += (self.tokens_per_page * (len(pages)-1))
            total_tokens += (self.tokens_per_page - self.page_space_left[pages[-1]])    
        return total_tokens

    
    def clear_cache(self):
        self.page_pool_keys.zero_()
        self.page_pool_values.zero_()
        self.page_table.clear()
        self.page_space_left.clear()
        self.empty_pages = [page for page in range(self.num_pages)]


    def reclaim(self, request_id, num_layers):
        pages_to_reclaim = []
        for layer in range(num_layers):
            pages = self.page_table[(request_id, layer)]
            pages_to_reclaim.extend(pages)
            self.page_table.pop((request_id, layer), None)

        for page in pages_to_reclaim:
            self.page_space_left.pop(page, None)
        self.empty_pages.extend(pages_to_reclaim)


    def get_size_bytes(self):
        return self.page_pool_keys.element_size() * self.page_pool_keys.nelement() + \
               self.page_pool_values.element_size() * self.page_pool_values.nelement()

   
   