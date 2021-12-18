import torchvision.transforms as standard_transforms
import torch
import cv2
from PIL import Image
from crowd_datasets.SHHA.loading_data import loading_data

a = torch.arange(9).view(3,3)
for i in a:
    i*=2

print(__file__ , __cached__ , __package__ , __name__)

img = cv2.imread("./faceScene.jpg")
print(img.shape)
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img = standard_transforms.ToTensor()(img)
print(img.shape)
