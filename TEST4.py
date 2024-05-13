import torch
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_gpu_2 = points.to(device='cuda')
print(points_gpu)
print(points)