
import torch.nn as nn
from utils.engine import *
import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import ModelEma, add_weight_decay
from randaugment import RandAugment
from src.models.build_model_TF import MyModel

import torch.nn as nn
from utils.LT_engine_grouplr import *
from src.data_loader.datasets import build_dataset
from src.loss_functions.dbl import *
from src.loss_functions.asl import *

from config import *

import clip


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', help='path to dataset', default='/data/yanjiexuan/coco')
parser.add_argument('--dataset', default='coco-lt', type=str, choices=['voc-lt', 'coco-lt'], help='dataset name')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--backbone', default='resnet101')
parser.add_argument('--pretrained', default='/data2/yanjiexuan/checkpoints/RC-Tran/pretrained_models/resnet101.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--threshold', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='the value of hyper-parameter alpha')
parser.add_argument('--drop', default=0.1, type=float,
                    help='the value of hyper-parameter drop-out')

parser.add_argument('--ctx_init', default='a photo of a', type=str, help='init context prompt')
parser.add_argument('--n_ctx', default=4, type=int, help='length M of context prompt when initializing')
parser.add_argument('--class_token_position', default='end', type=str, help='position of class token')

parser.add_argument('--loss_function', default='asl', type=str, choices=['asl', 'bce', 'dbl', 'mls', 'FL', 'CBloss', 'DBloss-noFocal', 'DBloss'], help='loss function')
parser.add_argument('--pretrain_clip', default='RN50', type=str, choices=['RN50', 'ViT16'], help='pretrained clip backbone')

def main_coco():
    args = parser.parse_args()

    if args.pretrain_clip == "RN50":
        pretrain_clip_path = 'data/huggingface/openai/pretrained/RN50.pt'
    elif args.pretrain_clip == "ViT16":
        pretrain_clip_path = 'data/huggingface/openai/pretrained/ViT-B-16.pt'

    print(f"Loading CLIP (backbone: {args.pretrain_clip})")
    clip_model, preprocess = clip.load(pretrain_clip_path, device='cpu', jit=False) # Must set jit=False for training


    # def convert_models_to_fp32(model): 
    #     for p in model.parameters(): 
    #         p.data = p.data.float() 
    #         p.grad.data = p.grad.data.float() 
    
    # clip.model.convert_weights(clip_model) # Actually this line is unnecessary since clip by default already on float16

    if args.dataset == 'coco-lt':
        dataset_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
            'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
            'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
            ]
        train_dataset = build_dataset(dataset=args.dataset, split='train')

        val_dataset = build_dataset(dataset=args.dataset, split='test')
    
    else:
        dataset_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'
            ]
        train_dataset = build_dataset(dataset=args.dataset, split='train')
        val_dataset = build_dataset(dataset=args.dataset, split='test')

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # loss functions
    if args.dataset == 'coco-lt':
        freq_file = './data/coco/class_freq.pkl'
    elif args.dataset == 'voc-lt':
        freq_file='./data/voc/class_freq.pkl'

    if args.loss_function == 'bce':
        loss_function = nn.BCEWithLogitsLoss()
    if args.loss_function == 'mls':
        loss_function = nn.MultiLabelSoftMarginLoss()
    if args.loss_function == 'FL':
        if args.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func=None,
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=1.0, freq_file=freq_file
            )
        elif args.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func=None,
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=1.0, freq_file=freq_file
            )
    if args.loss_function == 'CBloss': #CB
        if args.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='CB',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                loss_weight=10.0, freq_file=freq_file
            )
        elif args.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='CB',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                loss_weight=10.0, freq_file=freq_file
            )
    if args.loss_function == 'DBloss-noFocal': # DB-0FL
        if args.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=False, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=0.5, freq_file=freq_file
            )
        elif args.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=False, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=0.5, freq_file=freq_file
            )
    if args.loss_function == 'asl':
        loss_function = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    if args.loss_function == 'dbl':
        if args.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=1.0, freq_file=freq_file
            )
        elif args.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=1.0, freq_file=freq_file
            )


    # hyper-parameters
    args = {
        'backbone': args.backbone,
        'pretrained': args.pretrained,
        'num_classes': args.num_classes,
        'max_epoch': 120,
        'resume': args.resume,
        'evaluation': args.evaluate,
        'threshold': args.threshold,
        'dataset': args.dataset,
        'lr': args.lr,
        'alpha': args.alpha,
        'drop': args.drop,
        'train': True,
        'ctx_init': args.ctx_init,
        'n_ctx': args.n_ctx,
        'class_token_position': args.class_token_position
    }
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    
    # load model
    print('creating model {}'.format(model_name))
    models = nn.ModuleList([])
    if args['evaluation']:
        model = MyModel(args)
        models.append(model)
    else:
        regular_model = MyModel(args,classnames=dataset_classes, clip_model=clip_model).cuda()
        regular_model = torch.nn.DataParallel(regular_model)
        ema_model = ModelEma(regular_model, 0.998)  # 0.9997^641=0.82
        models.append(regular_model)
        models.append(ema_model)

    # print("Turning off gradients in both the image and the text encoder")
    # for name, param in models.named_parameters():
    #     if "text_encoder" in name:
    #         param.requires_grad = False



    # set optimizer
    Epochs = 80
    weight_decay = 1e-4
    # loss function
    criterion = nn.ModuleList([])
    criterion.append(loss_function)

    parameters = add_weight_decay(models[0], weight_decay)
    optimizer = torch.optim.AdamW(params=parameters, lr=args['lr'], weight_decay=0) # true wd, filter_bias_and_bn

    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args['lr'], steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.1)

    engine = MultiLabelEngine(args)
    engine.learning(models, train_loader, val_loader, criterion, optimizer, scheduler)


if __name__ == '__main__':
    main_coco()
