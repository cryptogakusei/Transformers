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

### Box for simple self-attention
class SimpleAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
            ) # creates an upper triangular matrix


    def forward(self, x):
        _, seq_len, _ = x.shape

        queries = self.W_q(x) # batch_size x seq_len x emb_dim
        keys = self.W_k(x) # -- same here --
        values = self.W_v(x) # -- same here --

        scores = queries @ keys.transpose(1,2) # batch_size x seq_len x seq_len
        scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf) # -- same here --
        weights = torch.softmax(scores / keys.shape[-1]**0.5, dim=-1) # -- same here --
        weights = self.dropout(weights) # -- same here --
        output = weights @ values
        return output


### Naive Multi-Head Attention box
class NaiveMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.heads = nn.ModuleList([
            SimpleAttention(d_in, d_out // num_heads, context_length, dropout, qkv_bias) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_in, d_out)
        

    def forward(self, x):
        return self.out_proj(torch.cat([head(x) for head in self.heads], dim=-1))


### MHA box
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(cfg["emb_dim"])
        # self.attention = SimpleAttention(
        #                     d_in=cfg["emb_dim"],
        #                     d_out=cfg["emb_dim"],
        #                     context_length=cfg["context_length"],
        #                     dropout=cfg["dropout_rate"],
        #                     qkv_bias=cfg["qkv_bias"],
        #                 )
        self.attention = NaiveMultiHeadAttention(
                            d_in=cfg["emb_dim"],
                            d_out=cfg["emb_dim"], 
                            context_length=cfg["context_length"], 
                            dropout=cfg["dropout_rate"], 
                            num_heads=cfg["num_heads"], 
                            qkv_bias=cfg["qkv_bias"]
                        )
        # self.attention = MultiHeadAttention(cfg)
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.layernorm2 = nn.LayerNorm(cfg["emb_dim"])
        self.feedforward = FeedForward(cfg)

    def forward(self, x):
        res = x
        x = self.layernorm1(x) # batch_size x seq_len x hidden_dim
        x = self.attention(x) # -- same here --
        x = self.dropout(x) # -- same here --
        x = x + res

        res = x
        x = self.layernorm2(x) # -- same here --
        x = self.feedforward(x) # -- same here --
        x = self.dropout(x) # -- same here --
        x = x + res

        return x


class MHAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg["dropout_rate"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.final_layernorm = nn.LayerNorm(cfg["emb_dim"])
        self.output_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        tok_embed = self.tok_emb(in_idx) # batch_size x seq_len x emb_dim
        pos_embed = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # seq_len x emb_dim
        x = tok_embed + pos_embed # batch_size x seq_len x emb_dim
        x = self.dropout_emb(x) # -- same here --
        x = self.blocks(x) # -- same here --
        x = self.final_layernorm(x) # -- same here --
        logits = self.output_layer(x) # batch_size x seq_len x vocab_size
        return logits
