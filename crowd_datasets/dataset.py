import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import torchvision.transforms as standard_transforms


class SHHATechA(Dataset):
    """data_root 目录下面存放2个list
    transform: totensor / Normalize
    train: train模式下， 缩放/裁剪/翻转
    patch: 对一张图片随机crop成patch的数量
    flip: 翻转
    """
    def __init__(self, data_root, transform=True, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "shanghai_tech_part_a_train.list"
        self.eval_list = "shanghai_tech_part_a_test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}   # 图片路径，标签路径
        self.img_list = []

        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        """"scale - crop - flip"""
        assert index <= len(self), "【%s】index over range!" % __file__
        img_path = self.img_list[index]
        label_path = self.img_map[img_path]
        img, point = self.load_data((img_path , label_path), self.train)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            assert min_size > 128 , "数据集中所有的图片最短边必须大于128"
            scale = random.uniform(*scale_range)
            # 确保缩放后图片的最短边仍然大于128
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale

        if self.train and self.patch:
            img, point = self.random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]
        
        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


    
    def load_data(self, img_gt_path, train):
        img_path, gt_path = img_gt_path
        # load the images
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # load ground truth points
        points = []
        with open(gt_path) as f_label:
            for line in f_label:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                points.append([x, y])

        return img, np.array(points)

    def random_crop(self, img, den, num_patch=4):
        half_h = 128
        half_w = 128
        result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
        result_den = []
        # crop num_patch for each image
        for i in range(num_patch):
            start_h = random.randint(0, img.size(1) - half_h)
            start_w = random.randint(0, img.size(2) - half_w)
            end_h = start_h + half_h
            end_w = start_w + half_w
            # copy the cropped rect
            result_img[i] = img[:, start_h:end_h, start_w:end_w]
            # copy the cropped points
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            # shift the corrdinates
            record_den = den[idx]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h

            result_den.append(record_den)

        return result_img, result_den


class SHHBTechB(Dataset):
    """data_root 目录下面存放2个list
    transform: totensor / Normalize
    train: train模式下， 缩放/裁剪/翻转
    patch: 对一张图片随机crop成patch的数量
    flip: 翻转
    """
    def __init__(self, data_root, transform=True, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "shanghai_tech_part_b_train.list"
        self.eval_list = "shanghai_tech_part_b_test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}   # 图片路径，标签路径
        self.img_list = []

        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        """"scale - crop - flip"""
        assert index <= len(self), "【%s】index over range!" % __file__
        img_path = self.img_list[index]
        label_path = self.img_map[img_path]
        img, point = self.load_data((img_path , label_path), self.train)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            assert min_size > 128 , "数据集中所有的图片最短边必须大于128"
            scale = random.uniform(*scale_range)
            # 确保缩放后图片的最短边仍然大于128
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale

        if self.train and self.patch:
            img, point = self.random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]
        
        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


    
    def load_data(self, img_gt_path, train):
        img_path, gt_path = img_gt_path
        # load the images
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # load ground truth points
        points = []
        with open(gt_path) as f_label:
            for line in f_label:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                points.append([x, y])

        return img, np.array(points)

    def random_crop(self, img, den, num_patch=4):
        half_h = 128
        half_w = 128
        result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
        result_den = []
        # crop num_patch for each image
        for i in range(num_patch):
            start_h = random.randint(0, img.size(1) - half_h)
            start_w = random.randint(0, img.size(2) - half_w)
            end_h = start_h + half_h
            end_w = start_w + half_w
            # copy the cropped rect
            result_img[i] = img[:, start_h:end_h, start_w:end_w]
            # copy the cropped points
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            # shift the corrdinates
            record_den = den[idx]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h

            result_den.append(record_den)

        return result_img, result_den


class NWPU(Dataset):
    """data_root 目录下面存放2个list
    transform: totensor / Normalize
    train: train模式下， 缩放/裁剪/翻转
    patch: 对一张图片随机crop成patch的数量
    flip: 翻转
    """
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "NWPU-CROWD-train.list"
        self.eval_list = "NWPU-CROWD-test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}   # 图片路径，标签路径
        self.img_list = []

        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])

        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        """"scale - crop - flip"""
        assert index <= len(self), "【%s】index over range!" % __file__
        img_path = self.img_list[index]
        label_path = self.img_map[img_path]
        img, point = self.load_data((img_path , label_path), self.train)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            assert min_size > 128 , "数据集中所有的图片最短边必须大于128"
            scale = random.uniform(*scale_range)
            # 确保缩放后图片的最短边仍然大于128
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale

        if self.train and self.patch:
            img, point = self.random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]
        
        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


    
    def load_data(self, img_gt_path, train):
        img_path, gt_path = img_gt_path
        # load the images
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # load ground truth points
        points = []
        with open(gt_path) as f_label:
            for line in f_label:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                points.append([x, y])
        assert len(points) > 0, "【%s contain no point!】" % img_path
        return img, np.array(points)

    def random_crop(self, img, den, num_patch=4):
        half_h = 128
        half_w = 128
        result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
        result_den = []
        # crop num_patch for each image
        for i in range(num_patch):
            start_h = random.randint(0, img.size(1) - half_h)
            start_w = random.randint(0, img.size(2) - half_w)
            end_h = start_h + half_h
            end_w = start_w + half_w
            # copy the cropped rect
            result_img[i] = img[:, start_h:end_h, start_w:end_w]
            # copy the cropped points
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            # shift the corrdinates
            record_den = den[idx]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h

            result_den.append(record_den)

        return result_img, result_den


class QNRF(Dataset):
    """data_root 目录下面存放2个list
    transform: totensor / Normalize
    train: train模式下， 缩放/裁剪/翻转
    patch: 对一张图片随机crop成patch的数量
    flip: 翻转
    """
    def __init__(self, data_root, transform=True, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "UCF-QNRF-train.list"
        self.eval_list = "UCF-QNRF-test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}   # 图片路径，标签路径
        self.img_list = []

        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        """"scale - crop - flip"""
        assert index <= len(self), "【%s】index over range!" % __file__
        img_path = self.img_list[index]
        label_path = self.img_map[img_path]
        img, point = self.load_data((img_path , label_path), self.train)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            assert min_size > 128 , "数据集中所有的图片最短边必须大于128"
            scale = random.uniform(*scale_range)
            # 确保缩放后图片的最短边仍然大于128
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale

        if self.train and self.patch:
            img, point = self.random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]
        
        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


    
    def load_data(self, img_gt_path, train):
        img_path, gt_path = img_gt_path
        # load the images
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # load ground truth points
        points = []
        with open(gt_path) as f_label:
            for line in f_label:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                points.append([x, y])

        return img, np.array(points)

    def random_crop(self, img, den, num_patch=4):
        half_h = 128
        half_w = 128
        result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
        result_den = []
        # crop num_patch for each image
        for i in range(num_patch):
            start_h = random.randint(0, img.size(1) - half_h)
            start_w = random.randint(0, img.size(2) - half_w)
            end_h = start_h + half_h
            end_w = start_w + half_w
            # copy the cropped rect
            result_img[i] = img[:, start_h:end_h, start_w:end_w]
            # copy the cropped points
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            # shift the corrdinates
            record_den = den[idx]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h

            result_den.append(record_den)

        return result_img, result_den


class UCFCC50(Dataset):
    """data_root 目录下面存放2个list
    transform: totensor / Normalize
    train: train模式下， 缩放/裁剪/翻转
    patch: 对一张图片随机crop成patch的数量
    flip: 翻转
    """
    def __init__(self, data_root, transform=True, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "UCF-QNRF-train.list"  # TODO
        self.eval_list = "UCF-QNRF-test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}   # 图片路径，标签路径
        self.img_list = []

        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        """"scale - crop - flip"""
        assert index <= len(self), "【%s】index over range!" % __file__
        img_path = self.img_list[index]
        label_path = self.img_map[img_path]
        img, point = self.load_data((img_path , label_path), self.train)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            assert min_size > 128 , "数据集中所有的图片最短边必须大于128"
            scale = random.uniform(*scale_range)
            # 确保缩放后图片的最短边仍然大于128
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale

        if self.train and self.patch:
            img, point = self.random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]
        
        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


    
    def load_data(self, img_gt_path, train):
        img_path, gt_path = img_gt_path
        # load the images
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # load ground truth points
        points = []
        with open(gt_path) as f_label:
            for line in f_label:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                points.append([x, y])

        return img, np.array(points)

    def random_crop(self, img, den, num_patch=4):
        half_h = 128
        half_w = 128
        result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
        result_den = []
        # crop num_patch for each image
        for i in range(num_patch):
            start_h = random.randint(0, img.size(1) - half_h)
            start_w = random.randint(0, img.size(2) - half_w)
            end_h = start_h + half_h
            end_w = start_w + half_w
            # copy the cropped rect
            result_img[i] = img[:, start_h:end_h, start_w:end_w]
            # copy the cropped points
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            # shift the corrdinates
            record_den = den[idx]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h

            result_den.append(record_den)

        return result_img, result_den



