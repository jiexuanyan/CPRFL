import os
import argparse
from PIL import Image

import torch.nn.functional as F
import torch

import clip

from randaugment import RandAugment

from config import *

from src.models.build_model_TF import MyModel

import torchvision.transforms as transforms

def main(args):

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Image Encoder
    if args.pretrain_clip == "RN50":
        pretrain_clip_path = '/data2/yanjiexuan/huggingface/openai/pretrained/RN50.pt'
    elif args.pretrain_clip == "ViT16":
        pretrain_clip_path = '/data2/yanjiexuan/huggingface/openai/pretrained/ViT-B-16.pt'

    print(f"Loading CLIP (backbone: {args.pretrain_clip})")
    clip_model, preprocess = clip.load(pretrain_clip_path, device='cpu', jit=False) # Must set jit=False for training


    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 
    
    clip.model.convert_weights(clip_model) # Actually this line is unnecessary since clip by default already on float16

    if args.dataset=='coco-lt':
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

    elif args.dataset=='voc-lt': 
        dataset_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'
            ]

    args = {
        'backbone': args.backbone,
        'pretrained': args.pretrained,
        'max_epoch': 150,
        'resume': args.resume,
        'evaluation': args.evaluate,
        # 'alpha': args.alpha,
        'drop':args.drop,
        'train': False,

        'dataset':args.dataset,
        # 'inp_seman':inp_seman,
        # 'data': args.data,
        # 'epis': args.epis,
        # 'shot': args.shot,
        'fix_sample':False,

        "head_num":head_num,
        "dim_head":dim_head,
        "feature_d":feature_d,
        # "use_cos":use_cos,
        "nonlinear":nonlinear,
        "out2_neck":out2_neck,
        "mlp_dim":mlp_dim,
        "expand":expand
    }

    regular_model = MyModel(args,classnames=dataset_classes, clip_model=clip_model).cuda()

    if args['dataset']=='voc-lt': 
        checkpoint = torch.load(args['resume'])
        filtered_dict = {k.replace("module.",""): v for k, v in checkpoint['state_dict'].items()}
        regular_model.load_state_dict(filtered_dict)
    elif args['dataset']=='coco-lt': 
        checkpoint = torch.load(args['resume'])
        filtered_dict = {k.replace("module.",""): v for k, v in checkpoint['state_dict'].items()}
        regular_model.load_state_dict(filtered_dict)
    

    # Preprocess Image
    transform = transforms.Compose([transforms.Resize((448,448)),
                                 transforms.ToTensor(),
                                        ])
    img = Image.open('/data2/yanjiexuan/coco/data/val2017/000000369751.jpg').convert('RGB')
    img = transform(img).unsqueeze(0)

    # Infer 
    with torch.no_grad():
        pred_score, _ = regular_model(img.to("cuda" if torch.cuda.is_available() else "cpu"))

    pred_score = torch.sigmoid(pred_score)

    # pred_score = pred_score / pred_score.norm(dim=-1, keepdim=True)

    print(pred_score)
    _, topk_preds = pred_score.topk(10)

    print("Top-10 Predictions: ")
    print(topk_preds)
    # for idx in topk_preds[0]:
    #     print(dataset_classes[idx].strip().ljust(20) + str(float(pred_score[:, idx].data))[:6])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help='dataset', default='coco-lt',type=str,choices=['coco-lt','voc-lt'])
    parser.add_argument('--backbone', default='resnet101')
    parser.add_argument('--pretrained', default='/data2/yanjiexuan/checkpoints/RC-Tran/pretrained_models/resnet101.pth', type=str)
    parser.add_argument('--resume', default='/data2/yanjiexuan/checkpoints/RC-Tran/LT_checkpoint/ema_Encoder_nonlinear_d8_2048_mlp2048_0.5x_lr5e-5_asl_clip_coco_best_66.691_e15.ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default = True,
                    help='evaluate model on validation set')

    parser.add_argument('--drop', default=0.6, type=float, help='dropout rate')

    parser.add_argument('--feature', default='', type=str)
    # parser.add_argument('--image_path', help='image path', default='/data2/yanjiexuan/voc/VOCdevkit/VOC2007/JPEGImages/000014.jpg')

    parser.add_argument('--pretrain_clip', default='RN50', type=str, choices=['RN50', 'ViT16'], help='pretrained clip backbone')

    args = parser.parse_args()

    main(args)