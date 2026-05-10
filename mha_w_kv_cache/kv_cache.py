import torch
import torch.nn as nn


### KV Cache - define it for one per layer
class KVCache:
    def __init__(self, context_length):
        self.K_cache = None
        self.V_cache = None
        self.context_length = context_length
    
    def cache(self, K_new, V_new):
        if self.K_cache is None:
            self.K_cache = K_new # batch_size x num_heads x 1 x head_dim
            self.V_cache = V_new # batch_size x num_heads x 1 x head_dim
        else:
            self.K_cache = torch.cat([self.K_cache, K_new], dim=2) # batch_size x num_heads x min(tokens_seen_so_far, context_length) x head_dim
            self.V_cache = torch.cat([self.V_cache, V_new], dim=2) # batch_size x num_heads x min(tokens_seen_so_far, context_length) x head_dim
        
        if self.K_cache.shape[2] > self.context_length:
            self.K_cache = self.K_cache[:,:, -self.context_length:, :] # cap at context_length
            self.V_cache = self.V_cache[:,:, -self.context_length:, :] 

    def get_cache(self):
        return self.K_cache, self.V_cache 
    
    def clear_cache(self):
        self.K_cache = None
        self.V_cache = None

    def get_size_bytes(self):
        if self.K_cache is None:
            return 0
        return self.K_cache.element_size() * self.K_cache.nelement() + self.V_cache.element_size() * self.V_cache.nelement()
    


### KV Cache - define it for one per layer
class OptimizedKVCache:
    # CAVEAT: built for batch size of 1 and for inserting 1 new token
    def __init__(self, context_length, max_seq_len_kv_cache, num_heads, head_dim):
        self.K_cache = torch.empty(1, num_heads, max_seq_len_kv_cache, head_dim)
        self.V_cache = torch.empty(1, num_heads, max_seq_len_kv_cache, head_dim)
        self.context_length = context_length
        self.pos = 0 # always indicates index where next token(s) were to be cached
        self.max_seq_len = max_seq_len_kv_cache

    def cache(self, K_new, V_new):
        # K_new and V_new are assumed to add only one new token in the sequence
        if self.pos >= self.max_seq_len:
            self.K_cache[:,:,:-1,:] = self.K_cache[:,:,1:,:] # the shifting of elements from 1 to max_seq_len_kv_cache-1 positions
            self.V_cache[:,:,:-1,:] = self.V_cache[:,:,1:,:]
            self.pos = self.max_seq_len - 1
        self.K_cache[:,:,self.pos,:] = K_new.squeeze(2) # dim(K_new) = 1 x num_heads x 1 x head_dim --> 1 x num_heads x head_dim --> insertion to K_cache
        self.V_cache[:,:,self.pos,:] = V_new.squeeze(2) # -- same as above --
        self.pos += 1

    def get_cache(self):
        start_pos = max(0, self.pos - self.context_length) # only send back the last context_length -- for orange-to-orange comparison of speedup with non KV cache
        return self.K_cache[:,:,start_pos:self.pos,:], self.V_cache[:,:,start_pos:self.pos,:] 
        
    def clear_cache(self):
        self.K_cache.zero_()
        self.V_cache.zero_()
        self.pos = 0

    def get_size_bytes(self):
        return self.K_cache.element_size() * self.K_cache.nelement() + \
               self.V_cache.element_size() * self.V_cache.nelement()

   