import torch
import torch.nn as nn


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
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
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
        

    def forward(self, x, return_attention=False):
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
        scores = queries @ keys.transpose(2, 3) # batch_size x num_heads x seq_len x seq_len
        scores = scores / keys.shape[-1]**0.5 # -- same here --
        scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf) # -- same here --
        attention = torch.softmax(scores, dim=-1) # -- same here --
        attention = self.dropout(attention) # -- same here --
        output = (attention @ values).transpose(1, 2) # batch_size x num_heads x seq_len x head_dim ---> # batch_size x seq_len x num_heads x head_dim
        output = output.contiguous().view(batch_size, seq_len, self.d_out) # batch_size x seq_len x d_out
        output = self.out_proj(output)

        if return_attention:
            return output, attention
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
                            qkv_bias=cfg["qkv_bias"]
                        )
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.layernorm2 = nn.LayerNorm(cfg["emb_dim"])
        self.feedforward = FeedForward(cfg)

    def forward(self, x, return_attention=False):
        res = x
        x = self.layernorm1(x) # batch_size x seq_len x hidden_dim
        if return_attention:
            x, attention = self.attention(x, return_attention) # -- same here --
        else:
            x = self.attention(x)
        x = self.dropout(x) # -- same here --
        x = x + res

        res = x
        x = self.layernorm2(x) # -- same here --
        x = self.feedforward(x) # -- same here --
        x = self.dropout(x) # -- same here --
        x = x + res
        
        if return_attention:
            return x, attention
        return x

class MHAModelInterpretability(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg["dropout_rate"])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.final_layernorm = nn.LayerNorm(cfg["emb_dim"])
        self.output_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, in_idx, return_attention=False):
        _, seq_len = in_idx.shape
        tok_embed = self.tok_emb(in_idx) # batch_size x seq_len x emb_dim
        pos_embed = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # seq_len x emb_dim
        x = tok_embed + pos_embed # batch_size x seq_len x emb_dim
        x = self.dropout_emb(x) # -- same here --

        all_attention = []
        for block in self.blocks:
            if return_attention:
                x, attention = block(x, return_attention) # -- same here --
                all_attention.append(attention)
            else:
                x = block(x) # -- same here --
        x = self.final_layernorm(x) # -- same here --
        logits = self.output_layer(x) # batch_size x seq_len x vocab_size

        if return_attention:
            return logits, all_attention
        return logits
