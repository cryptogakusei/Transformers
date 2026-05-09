import torch
x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(x)
x_pairs = x.view(2, -1, 2)      

print(x_pairs)
print(-x_pairs[:, :, 1])
x_rotated = torch.stack([-x_pairs[:, :, 1], x_pairs[:, :, 0]], dim=-1)
print(x_rotated)
x_rotated = x_rotated.flatten(start_dim=-2, end_dim=-1)   
print(x_rotated)