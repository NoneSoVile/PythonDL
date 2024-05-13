from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import  torch
import json, os
import requests
import requests

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

shape = points.shape

print(shape)
batch_t = torch.randn(2, 3, 5, 5)
img_t = torch.randn(3, 5, 5) # shape [channels, rows, columns]
weights = torch.tensor([0.2126, 0.7152, 0.0722])

img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
print(img_gray_naive.shape, batch_gray_naive.shape)
