import argparse
import datetime
import random
import os
import time
from pathlib import Path
import torch

import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets.dataset import QNRF, SHHATechA, NWPU, SHHBTechB
from engine import *
from models.p2pnet import build_model
from models.p2pnet import build_criterion

from zcdebugtool import watch_model_para

# from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():

    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,help='gradient clipping max norm')

    # Model parameters 
    parser.add_argument('--frozen_weights', type=str, default=None,help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument('--cls_loss_coef', default=1., type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float,help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHATechA')
    parser.add_argument('--data_root', default='./data/SHHATechA/',help='path where the dataset is')
    parser.add_argument('--output_dir', default='./output_dir',help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./checkpoints_dir',help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./tensorboard_dir',help='path where to save, empty for no saving')

    # 
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--eval_freq', default=1, type=int,help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=3, type=int, help='the gpu used for training')

    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    # backup the arguments
    print(args)

    device = torch.device('cuda')
    print(torch.cuda.get_device_capability())
    print(torch.cuda.get_device_name())

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # get the P2PNet model
    model = build_model(args)
    criterion = build_criterion(args)

    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    watch_model_para(model_without_ddp)

    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    assert args.dataset_file in ["SHHATechA", "NWPU", "QNRF", "SHHBTechB"], "dataset_file not in [\"SHHATechA\", \"NWPU\", \"QNRF\"]"

    # create the dataset
    # create the training and valiation set
    if args.dataset_file in "SHHATechA":
        train_set = SHHATechA(data_root = args.data_root, train=True, patch=True, flip=True)
        val_set = SHHATechA(data_root = args.data_root, train=False, patch=False, flip=False)
    elif args.dataset_file in "SHHBTechB":
        train_set = SHHBTechB(data_root = args.data_root, train=True, patch=True, flip=True)
        val_set = SHHBTechB(data_root = args.data_root, train=False, patch=False, flip=False)
    elif args.dataset_file in "NWPU":
        train_set = NWPU(data_root = args.data_root, train=True, patch=True, flip=True)
        val_set = NWPU(data_root = args.data_root, train=False, patch=False, flip=False)
    elif args.dataset_file in "QNRF":
        train_set = QNRF(data_root = args.data_root, train=True, patch=True, flip=True)
        val_set = QNRF(data_root = args.data_root, train=False, patch=False, flip=False)

    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)
    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    mae = []
    mse = []
    bestmae = 100000
    print("【***************Start Training****************】")

    for epoch in range(args.start_epoch, args.epochs):
        print("【**********Training  {} **********】".format(epoch))
        stat = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)

        # change lr according to the scheduler
        lr_scheduler.step()

        torch.save({'model': model_without_ddp.state_dict(),}, os.path.join(args.checkpoints_dir, 'latest.pth'))

        if epoch % args.eval_freq == 0 and epoch > 0:
            print("【Evaluating  {} **********】".format(epoch))
            result = evaluate_crowd_crop_x(model, data_loader_val, device)
            mae.append(result[0])
            mse.append(result[1])
            if bestmae > result[0]:
                bestmae = result[0]
            print("MAE: {}\t MSE: {}\t BestMAE: {}\n ".format(result[0], result[1], bestmae))
            if abs(np.min(mae) - result[0]) < 0.01:
                torch.save({'model': model_without_ddp.state_dict(),}, os.path.join(args.checkpoints_dir, 'best_mae_{}.pth'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)