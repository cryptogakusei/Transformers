import torch
import torch.nn as nn

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
            self.K_cache[:,:,:-1,:] = self.K_cache[:,:,1:,:]
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

    

### Feedforward box
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feedforward = nn.Sequential(
                            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                            nn.GELU(),
                            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.feedforward(x)

### An optimized Multi-head attention box
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, max_seq_len_kv_cache, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
            ) # creates an upper triangular matrix
        self.kv_cache = OptimizedKVCache(context_length, max_seq_len_kv_cache, self.num_heads, self.head_dim)

    def forward(self, x):
        batch_size, seq_len, d_in = x.shape # batch_size x 1 x d_in

        query = self.W_q(x) # batch_size x 1 x d_out
        key = self.W_k(x) # -- same here --
        value = self.W_v(x) # -- same here --

        # break the last dimension across heads
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim) # batch_size x 1 x num_heads x head_dim
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim) # batch_size x 1 x num_heads x head_dim
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim) # batch_size x 1 x num_heads x head_dim
        
        # prepare the matrices for matmul by rearranging dimensions
        key = key.transpose(1, 2) # batch_size x 1 x num_heads x head_dim ---> # batch_size x num_heads x 1 x head_dim
        value = value.transpose(1, 2) # batch_size x 1 x num_heads x head_dim ---> # batch_size x num_heads x 1 x head_dim
        query = query.transpose(1, 2) # batch_size x 1 x num_heads x head_dim ---> # batch_size x num_heads x 1 x head_dim

        # cache in the KV_cache
        self.kv_cache.cache(key, value)

        # retrieve from the KV cache
        keys, values = self.kv_cache.get_cache() # both batch_size x num_heads x min(tokens_seen_so_far, context_length) x head_dim

        attention = query @ keys.transpose(2, 3) # batch_size x num_heads x 1 x min(tokens_seen_so_far, context_length)
        attention = attention / keys.shape[-1]**0.5 # -- same here --
        attention = torch.softmax(attention, dim=-1) # -- same here --
        attention = self.dropout(attention) # -- same here --
        output = (attention @ values).transpose(1, 2) # batch_size x num_heads x 1 x head_dim ---> # batch_size x 1 x num_heads x head_dim
        output = output.contiguous().view(batch_size, seq_len, self.d_out) # batch_size x 1 x d_out
        output = self.out_proj(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(cfg["emb_dim"])
        self.attention = MultiHeadAttention(
                            d_in=cfg["emb_dim"],
                            d_out=cfg["emb_dim"], 
                            context_length=cfg["context_length"], 
                            dropout=cfg["dropout_rate"], 
                            num_heads=cfg["num_heads"],
                            max_seq_len_kv_cache = cfg["max_seq_len_kv_cache"], 
                            qkv_bias=cfg["qkv_bias"]
                        )
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.layernorm2 = nn.LayerNorm(cfg["emb_dim"])
        self.feedforward = FeedForward(cfg)

    def forward(self, x):
        res = x
        x = self.layernorm1(x) # batch_size x 1 x hidden_dim
        x = self.attention(x) # -- same here --
        x = self.dropout(x) # -- same here --
        x = x + res

        res = x
        x = self.layernorm2(x) # -- same here --
        x = self.feedforward(x) # -- same here --
        x = self.dropout(x) # -- same here --
        x = x + res

        return x


class MHAModelOptimizedKV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg["dropout_rate"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.final_layernorm = nn.LayerNorm(cfg["emb_dim"])
        self.output_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])
        self.tokens_seen = 0
        self.context_length = cfg["context_length"]

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        tok_embed = self.tok_emb(in_idx) # batch_size x 1 x emb_dim
        position = self.tokens_seen % self.context_length        
        pos_embed = self.pos_emb(torch.tensor([position], device=in_idx.device)) # 1 x emb_dim
        self.tokens_seen += 1

        x = tok_embed + pos_embed # batch_size x 1 x emb_dim
        x = self.dropout_emb(x) # -- same here --
        x = self.blocks(x) # -- same here --
        x = self.final_layernorm(x) # -- same here --
        logits = self.output_layer(x) # batch_size x 1 x vocab_size
        return logits


    def clear_cache(self):
        self.tokens_seen = 0
        for block in self.blocks:
            block.attention.kv_cache.clear_cache()

    def get_total_kv_cache_size(self):
        total = 0
        for block in self.blocks:
            total += block.attention.kv_cache.get_size_bytes()
        return total
    
    