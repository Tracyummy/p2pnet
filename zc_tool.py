from sys import dont_write_bytecode
import cv2
import torch
import torchvision.transforms as standard_transforms
import PIL.Image as Image
import math
import torch.nn.functional as F


def watch_model_para(model):
    print('model_named_parameters:',[n for n, p in model.named_parameters()])
    # print('model_parameters:',[p for p in model.parameters()])


# data op:
def img_aug():
    img_path = "data/SHHATechA/train/IMG_1.jpg"
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])(img)


if __name__=='__main__':
    
    img_path = "data/SHHATechA/train/IMG_4.jpg"
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])(img)
    img = img.unsqueeze(0)
    h,w = img.shape[2:]
    newh = math.ceil(h/512) * 512
    neww = math.ceil(w/512) * 512
    diffh, diffw = newh - h, neww - w
    leftpad, rightpad = diffw//2, diffw - diffw//2
    toppad, bottompad = diffh//2, diffh - diffh//2

    img = F.pad(img, (leftpad, rightpad, toppad, bottompad), "constant", 0)
    print(img.shape)
    for i in range(0, newh, 512):
        for j in range(0, neww, 512):
            tmpimg = img[i:i+512, j:j+512]
            # outputs = model(tmpimg)
            # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            # outputs_points = outputs['pred_points'][0]
            # predict_cnt = int((outputs_scores > threshold).sum())

    print(img.shape, newh, neww)