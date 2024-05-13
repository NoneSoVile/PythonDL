from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import  torch
import json, os
import requests
local_file_path = "imagenet_class_index.json"
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

# Check if the local file exists
if os.path.exists(local_file_path):
    with open(local_file_path, "r") as f:
        data = json.load(f)
    print("Loaded JSON data from local file")
else:
    response = requests.get(url)
    data = response.json()
    with open(local_file_path, "w") as f:
        json.dump(data, f)
    print("Fetched JSON data from URL and saved to local file")

# Process the data as needed
imagenet_classes = [(int(key), value[1]) for key, value in data.items()]

np.set_printoptions(suppress=True)
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)])
#print(dir(models))

resnet = models.resnet101(pretrained=True)
#print(resnet)

img = Image.open("cat2.jpeg")
#img.show()
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
print(resnet.eval())
out = resnet(batch_t)
#with open('imagenet_classes.txt') as f:
#    labels = [line.strip() for line in f.readlines()]
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(imagenet_classes[index[0]], percentage[index[0]].item())