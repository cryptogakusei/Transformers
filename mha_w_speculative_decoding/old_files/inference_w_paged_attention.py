import torch
import torch.nn as nn

from positional_embedding import RoPE
from kv_cache import KVCache
    

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
    def __init__(self, layer, d_in, d_out, max_seq_len, dropout, num_heads, kvcache_limit, rope_limit, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.layer = layer
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.pos = 0 # needed for coutning how many tokens have been seen so as to be able to deduce the position of incoming token
        self.rope = RoPE(head_dim=self.head_dim, rope_limit=rope_limit)

    def forward(self, x, partition, active_requests, page_allocator):
        batch_size, seq_len, d_in = x.shape # batch_size x seq_len x d_in

        queries = self.W_q(x) # batch_size x seq_len x d_out
        keys = self.W_k(x) # -- same here --
        values = self.W_v(x) # -- same here --

        # break the last dimension across heads
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim) # batch_size x seq_len x num_heads x head_dim
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim) # batch_size x seq_len x num_heads x head_dim
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim) # batch_size x seq_len x num_heads x head_dim
        
        # prepare the matrices for matmul by rearranging dimensions
        keys = keys.transpose(1, 2) # batch_size x seq_len x num_heads x head_dim ---> # batch_size x num_heads x seq_len x head_dim
        values = values.transpose(1, 2) # batch_size x seq_len x num_heads x head_dim ---> # batch_size x num_heads x seq_len x head_dim
        queries = queries.transpose(1, 2) # batch_size x seq_len x num_heads x head_dim ---> # batch_size x num_heads x seq_len x head_dim

        expanded_output = None

        # Note the separation ofmask per requests that we did is great for GPU with fused kernels as we 
        # can take advattnage of parallel threads in GPU. But is worse in pytorch.

        # slicing have to be done for kv cache update and to apply rope
        for seq_range, request_index in partition.items():
            (start_pos, end_pos) = seq_range # end_pos is not inclusive, start_pos is inclusive
            request = active_requests[request_index]
            request_id = request.request_id
            
            # apply RoPE
            queries[:,:,start_pos:end_pos,:] = self.rope.apply_rope(queries[:,:,start_pos:end_pos,:], pos=request.pos) # batch_size x num_heads x (end_pos - start_pos) x head_dim
            keys[:,:,start_pos:end_pos,:] = self.rope.apply_rope(keys[:,:,start_pos:end_pos,:], pos=request.pos) # batch_size x num_heads x (end_pos - start_pos) x head_dim
            if self.layer == 0:
                request.pos += (end_pos - start_pos) # update to the next RoPE position for that active requests and we want to update it only once for each request in the bath of inference

            # cache in the KV_cache
            page_allocator.allocate(
                new_keys=keys[:,:,start_pos:end_pos,:], 
                new_values=values[:,:,start_pos:end_pos,:], 
                layer=self.layer,
                request_id=request_id)

            # retrieve from the KV cache
            slice_keys, slice_values = page_allocator.get_cache(layer=self.layer, request_id=request_id) # both batch_size x num_heads x min(tokens_seen_so_far, context_length) x head_dim
            slice_queries = queries[:,:,start_pos:end_pos,:]

            # construct the mask
            num_rows = end_pos - start_pos
            num_cols = slice_keys.shape[2]
            mask = torch.ones((num_rows, num_cols), dtype=torch.bool, device=queries.device)
            mask_row_identifier = []
            mask_col_identifier = []
            for row in range(num_rows):
                for col in range(num_cols - num_rows + 1 + row):
                    mask_row_identifier.append(row)
                    mask_col_identifier.append(col)
            mask[mask_row_identifier, mask_col_identifier] = False

            attention = slice_queries @ slice_keys.transpose(2, 3) # batch_size x num_heads x 1 x min(tokens_seen_so_far, context_length)
            attention = attention / slice_keys.shape[-1]**0.5 # -- same here --
            attention.masked_fill_(mask, -torch.inf) # -- same here --
            attention = torch.softmax(attention, dim=-1) # -- same here --
            attention = self.dropout(attention) # -- same here --
            output = (attention @ slice_values).transpose(1, 2) # batch_size x num_heads x 1 x head_dim ---> # batch_size x 1 x num_heads x head_dim
            output = output.contiguous().view(batch_size, end_pos-start_pos, self.d_out) # batch_size x 1 x d_out
            expanded_output = output if expanded_output is None else torch.cat([expanded_output,output], dim=1)

        expanded_output = self.out_proj(expanded_output)
        return expanded_output


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(cfg["emb_dim"])
        self.attention = MultiHeadAttention(
                            layer=layer,
                            d_in=cfg["emb_dim"],
                            d_out=cfg["emb_dim"], 
                            max_seq_len=cfg["max_seq_len"], 
                            dropout=cfg["dropout_rate"], 
                            num_heads=cfg["num_heads"],
                            rope_limit=cfg["rope_limit"], 
                            kvcache_limit=cfg["kvcache_limit"],
                            qkv_bias=cfg["qkv_bias"],
                        )
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.layernorm2 = nn.LayerNorm(cfg["emb_dim"])
        self.feedforward = FeedForward(cfg)

    def forward(self, x, partition, active_requests, page_allocator):
        res = x
        x = self.layernorm1(x) # batch_size x 1 x hidden_dim
        x = self.attention(x, partition, active_requests, page_allocator) # -- same here --
        x = self.dropout(x) # -- same here --
        x = x + res

        res = x
        x = self.layernorm2(x) # -- same here --
        x = self.feedforward(x) # -- same here --
        x = self.dropout(x) # -- same here --
        x = x + res

        return x


class MHAModelPagedAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg["dropout_rate"])
        self.blocks = nn.ModuleList([TransformerBlock(cfg, layer) for layer in range(cfg["num_layers"])])
        self.final_layernorm = nn.LayerNorm(cfg["emb_dim"])
        self.output_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, in_idx, partition, active_requests, page_allocator):
        x = self.tok_emb(in_idx) # batch_size x 1 x emb_dim
        x = self.dropout_emb(x) # -- same here --
        for block in self.blocks:
            x = block(x, partition, active_requests, page_allocator) # -- same here --
        x = self.final_layernorm(x) # -- same here --
        logits = self.output_layer(x) # batch_size x 1 x vocab_size
        return logits


    def clear_cache(self):
        for block in self.blocks:
            block.attention.kv_cache.clear_cache()
            block.attention.pos = 0

    def get_total_kv_cache_size(self):
        total = 0
        for block in self.blocks:
            total += block.attention.kv_cache.get_size_bytes()
        return total
    
    