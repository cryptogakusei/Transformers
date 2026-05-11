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
            self.K_cache = K_new # batch_size x num_heads x seq_len x head_dim, seq_len = max_num_batched_tokens in chunked_prefill phase
            self.V_cache = V_new # -- same here --
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
    def __init__(self, context_length, num_heads, head_dim, kvcache_limit=4096):
        self.K_cache = torch.empty(1, num_heads, kvcache_limit, head_dim)
        self.V_cache = torch.empty(1, num_heads, kvcache_limit, head_dim)
        self.context_length = context_length
        self.pos = 0 # always indicates index where next token(s) were to be cached
        self.kvcache_limit = kvcache_limit

    def cache(self, K_new, V_new):
        seq_len = K_new.shape[2]
        if self.pos + seq_len >= self.kvcache_limit:
            self.K_cache[:,:,:-seq_len,:] = self.K_cache[:,:,seq_len:,:] # the shifting for making space
            self.V_cache[:,:,:-seq_len,:] = self.V_cache[:,:,seq_len:,:]
            self.pos = self.kvcache_limit - 1
        self.K_cache[:,:,self.pos:self.pos + seq_len,:] = K_new # dim(K_new) = 1 x num_heads x seq_len x head_dim --> insertion to K_cache
        self.V_cache[:,:,self.pos:self.pos + seq_len,:] = V_new # -- same as above --
        self.pos += seq_len

    def get_cache(self):
        return self.K_cache, self.V_cache
        
    def clear_cache(self):
        self.K_cache.zero_()
        self.V_cache.zero_()
        self.pos = 0

    def get_size_bytes(self):
        return self.K_cache.element_size() * self.K_cache.nelement() + \
               self.V_cache.element_size() * self.V_cache.nelement()

   