import torch
import torch.nn as nn


### Applying RoPE
class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=4096, base=10000):
        super().__init__()
        power = torch.arange(head_dim // 2) # 1 x head_dim/2
        theta = base ** (- 2 * power / head_dim ) # 1 x head_dim/2 (this is theta_1, theta_2, ..., theta_d/2 from RoPE paper)
        theta_interleave = theta.repeat_interleave(2).unsqueeze(0) # 1 x head_dim
        position = torch.arange(max_seq_len).unsqueeze(1) # context_length x 1
        angles =  position * theta_interleave # context_length x head_dim

        self.register_buffer('cos', torch.cos(angles)) # context_length x head_dim
        self.register_buffer('sin', torch.sin(angles)) # context_length x head_dim       



    def apply_rope(self, x, pos=0):
        batch_size, num_head, seq_len, head_dim = x.shape 
        x_pairs = x.view(batch_size, num_head, seq_len, -1, 2) # batch_size x num_head x seq_len x head_dim --> batch_size x num_head x seq_len x head_dim/2 x 2 
        x_rotated = torch.stack([-x_pairs[..., 1], x_pairs[..., 0]], dim=-1)
        x_rotated = x_rotated.flatten(start_dim=-2, end_dim=-1) # batch_size x num_head x seq_len x head_dim
        return x * self.cos[pos:pos+seq_len, :] + x_rotated * self.sin[pos:pos+seq_len, :] # # batch_size x num_heads x seq_len x head_dim

