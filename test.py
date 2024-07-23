from audioop import avg
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from utils.LT_engine_test import MultiLabelEngine
# from src.loss_functions.losses import AsymmetricLoss, CosLoss
from config import *
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
from torch.cuda.amp import GradScaler, autocast

from PIL import Image
from src.data_loader.datasets import build_dataset
import tqdm
from src.helper_functions.metrics import *

# from src.models.model_init_text import MyModel
from src.models.model_coop import MyModel
# from src.models.model_visual_fine import MyModel
# from src.models.model_cocoop import MyModel

import clip
# use_cos = True

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--dataset', help='dataset', default='coco-lt',type=str,choices=['coco-lt','voc-lt'])
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--backbone', default='resnet101')
parser.add_argument('--pretrained', default='/data2/yanjiexuan/checkpoints/RC-Tran/pretrained_models/resnet101.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--threshold', default=0.5, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default = True,
                    help='evaluate model on validation set')
# parser.add_argument('--alpha', default=1, type=float,
#                     help='the value of hyper-parameter alpha')
parser.add_argument('--drop', default=0.6, type=float, help='dropout rate')
# parser.add_argument('--shot', default=5, type=int, choices=[1,5], help='1 or 5 shot')
# parser.add_argument('--epis', nargs = '+', type=str, default=['1','2','3','4','5','6','7','8','9','10'])
# parser.add_argument('--epis', nargs = '+', type=str, default=['8','9','10'])
parser.add_argument('--feature', default='', type=str)
parser.add_argument('--emb', help='word2vec', default='glove')

parser.add_argument('--pretrain_clip', default='RN50', type=str, choices=['RN50', 'ViT16'], help='pretrained clip backbone')

parser.add_argument('--ctx_init', default='a photo of a', type=str, help='init context prompt')
parser.add_argument('--n_ctx', default=4, type=int, help='length M of context prompt when initializing')
parser.add_argument('--class_token_position', default='end', type=str, help='position of class token')

def main():
    args = parser.parse_args()
    feature = args.feature
    # hyper-parameters
    # if args.backbone == 'resnet101':
    #     from src.models.build_model_TF import MyModel
    # elif args.backbone == 'googlenetv3':
    #     from src.models.build_model_g_TF import MyModel
    # elif args.backbone == 'fc':
    #     from src.models.build_model_fc import MyModel

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
        val_dataset = build_dataset(dataset=args.dataset, split='test', inp_name='/home/yanjiexuan/multi-label-fsl/RC-Tran-LT/data/coco/coco_glove_300_coco_sequence.pkl')

    elif args.dataset=='voc-lt': 
        dataset_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'
            ]
        val_dataset = build_dataset(dataset=args.dataset, split='test', inp_name='/home/yanjiexuan/multi-label-fsl/RC-Tran-LT/data/voc/voc_glove_word2vec.pkl')

    args = {
        'backbone': args.backbone,
        'pretrained': args.pretrained,
        'num_classes': args.num_classes,
        'max_epoch': 150,
        'resume': args.resume,
        'evaluation': args.evaluate,
        'threshold': args.threshold,
        'lr': args.lr,
        # 'alpha': args.alpha,
        'drop':args.drop,
        'train': False,

        'dataset':args.dataset,
        # 'inp_seman':inp_seman,
        # 'data': args.data,
        'image_size':args.image_size,
        'workers':args.workers,
        'batch_size': args.batch_size,
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
        "expand":expand,
        'ctx_init': args.ctx_init,
        'n_ctx': args.n_ctx,
        'class_token_position': args.class_token_position
    }

    print(args)
    # Setup model
    print('creating model...')

    regular_model = MyModel(args,classnames=dataset_classes, clip_model=clip_model).cuda()
    # regular_model = torch.nn.DataParallel(regular_model)
    # if args['resume']:
    #     if os.path.isfile(args['resume']):
    #         print("=> loading checkpoint '{}'".format(args['resume']))
    #         checkpoint = torch.load(args['resume'])
    #         filtered_dict = {k.replace("module.",""): v for k, v in checkpoint['state_dict'].items()}
    #         regular_model.load_state_dict(filtered_dict)
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args['resume']))

    # model = torch.nn.DataParallel(model)
    engine = MultiLabelEngine(args)

    if args['dataset']=='voc-lt': 
        checkpoint = torch.load(args['resume'])
        filtered_dict = {k.replace("module.",""): v for k, v in checkpoint['state_dict'].items()}
        regular_model.load_state_dict(filtered_dict)
    elif args['dataset']=='coco-lt': 
        checkpoint = torch.load(args['resume'])
        filtered_dict = {k.replace("module.",""): v for k, v in checkpoint['state_dict'].items()}
        regular_model.load_state_dict(filtered_dict)
    
    test_loader = torch.utils.data.DataLoader(
                                            val_dataset,
                                            batch_size=args['batch_size'],
                                            shuffle=False,
                                            num_workers = args['workers'],
                                            drop_last=False
                                        )

    regular_ap, regular_map, reg_meters, reg_topk= engine.learning(regular_model, test_loader)

    print("test mAP:{}".format(regular_map))
    head_AP, middle_AP, tail_AP, head, medium, tail = ltAnalysis(regular_ap, args['dataset'])
    filename = os.path.join('log/log_rebuttal', str(args['dataset'])+"_"+args['resume'][args['resume'].rfind("/")+1:].replace(".ckpt","_{}_{}.txt".format(feature, regular_map)))
    with open(filename,'a') as f:
        f.write(str(args)+"\n")
        f.write("=================================================>>>>>>> OP, OR, OF1, CP, CR, CF1:"+"\n")
        f.write(str(reg_meters)+"\n")
        f.write("=================================================>>>>>>> OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k:"+"\n")
        f.write(str(reg_topk)+"\n")
        f.write("test mAP:"+str(regular_map)+"\n")
        f.write("head APs:"+str(head_AP)+"\n")
        f.write("middle APs:"+str(middle_AP)+"\n")
        f.write("tail APs:"+str(tail_AP)+"\n")
        f.write("=================================================>>>>>>> mAP head, mAP medium, mAP tail:"+"\n")
        f.write(str(head)+","+str(medium)+","+str(tail)+"\n")


    
if __name__ == '__main__':
    main()
