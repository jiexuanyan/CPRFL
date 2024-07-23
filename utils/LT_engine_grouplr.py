import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from .util import *
from .util import AveragePrecisionMeter
from torch.cuda.amp import GradScaler, autocast
import torchnet as tnt
from config import *

from src.helper_functions.metrics import *

class MultiLabelEngine():
    def __init__(self, args):
        # hyper-parameters
        self.evaluation = args['evaluation']
        self.threshold = args['threshold']
        self.resume = args['resume']
        self.max_epoch = args['max_epoch']
        self.start_lr = args['lr']
        self.dataset = args['dataset']

        # measure mAP
        self.regular_ap_meter = AveragePrecisionMeter(threshold=self.threshold, difficult_examples=False)
        self.ema_ap_meter = AveragePrecisionMeter(threshold=self.threshold, difficult_examples=False)
        self.regular_loss_meter = tnt.meter.AverageValueMeter()
        self.ema_loss_meter = tnt.meter.AverageValueMeter()

        # best model
        self.highest_regular_map = 0
        self.highest_ema_map = 0
        self.highest_regular_model = None
        self.highest_ema_model = None


    def meter_reset(self):
        self.regular_ap_meter.reset()
        self.ema_ap_meter.reset()
        self.regular_loss_meter.reset()
        self.ema_loss_meter.reset()


    def meter_print(self):
        regular_loss = self.regular_loss_meter.value()[0]
        regular_ap = 100 * self.regular_ap_meter.value()
        regular_map = regular_ap.mean()
        OP, OR, OF1, CP, CR, CF1 = self.regular_ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.regular_ap_meter.overall_topk(3)
        print('=================================================>>>>>>> Experimental Results on regular {}'.format(model_name))
        print('mAP score: {map:.3f}\t loss: {loss:.3f}'.format(map=regular_map, loss=regular_loss))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR,
                                      CF1=CF1))
        print('OP_3: {OP:.4f}\t'
              'OR_3: {OR:.4f}\t'
              'OF1_3: {OF1:.4f}\t'
              'CP_3: {CP:.4f}\t'
              'CR_3: {CR:.4f}\t'
              'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k,
                                        CR=CR_k, CF1=CF1_k))

        ema_loss_meter = self.ema_loss_meter.value()[0]
        ema_ap = 100 * self.ema_ap_meter.value()
        ema_map = ema_ap.mean()
        OP, OR, OF1, CP, CR, CF1 = self.ema_ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ema_ap_meter.overall_topk(3)
        print('#################################################>>>>>>> Experimental Results on EMA {}'.format(model_name))
        print('mAP score: {map:.3f}\t loss: {loss:.3f}'.format(map=ema_map, loss=ema_loss_meter))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR,
                                      CF1=CF1))
        print('OP_3: {OP:.4f}\t'
              'OR_3: {OR:.4f}\t'
              'OF1_3: {OF1:.4f}\t'
              'CP_3: {CP:.4f}\t'
              'CR_3: {CR:.4f}\t'
              'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k,
                                        CR=CR_k, CF1=CF1_k))
        return regular_ap, regular_map, ema_ap, ema_map

    def learning(self, models, train_loader, val_loader, criterion, optimizer, scheduler):
        # optionally resume from a checkpoint
        self.start_epoch = 0
        if self.resume != '':
            if os.path.isdir(self.resume):
                if os.path.isfile(os.path.join(self.resume,'regular_checkpoint_cur.ckpt')):
                    print("=> loading checkpoint '{}'".format(self.resume))
                    checkpoint = torch.load(os.path.join(self.resume,'regular_checkpoint_cur.ckpt'))
                    self.start_epoch = checkpoint['epoch']
                    self.highest_regular_map = checkpoint['score']
                    models[0].load_state_dict(checkpoint['state_dict'])
                    
                    checkpoint = torch.load(os.path.join(self.resume,'EMA_checkpoint_cur.ckpt'))
                    self.highest_ema_map = checkpoint['score']
                    models[1].module.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.resume, self.start_epoch))

                    print('update scheduler, totoal {} x {} steps'.format(self.start_epoch+1,len(train_loader)))
                    for _ in range((self.start_epoch+1)*len(train_loader)):
                        scheduler.step()
                else:
                    print("=> no checkpoint found at '{}'".format(self.resume))
            else:
                print("=> no checkpoint found at '{}'".format(self.resume))


        if self.evaluation:
            models[0].eval()
            self.meter_reset()
            self.validate(models, val_loader, criterion)
            _, _ = self.meter_print()
        else:
            scaler = GradScaler()
            for epoch in range(self.start_epoch+1, self.max_epoch):
                # if epoch == 30:# for voc
                if epoch == 80:#for coco and nus-wide
                    assert 1 == 2
                # train step
                models[0].train()
                self.meter_reset()
                self.train(models, train_loader, criterion, optimizer, scheduler, scaler, epoch)
                # _, _ = self.meter_print()
                if epoch >10: #for coco and nus-wide
                # if epoch >20: #for voc 2007
                # evaluate step
                    models[0].eval()
                    self.meter_reset()
                    self.validate(models, val_loader, criterion)
                    regular_ap, regular_map, ema_ap, ema_map = self.meter_print()
                    self.save_checkpoint(models, regular_ap, regular_map, ema_ap, ema_map, epoch)
                    models[0].train()

    def validate(self, models, val_loader, criterion):
        regular_model = models[0]
        ema_model = models[1]
        val_loader = tqdm(val_loader, desc='Test')
        for i, (inputData, target, semantic) in enumerate(val_loader):
            # compute output
            target = target.cuda()
            semantic = semantic[0].cuda().float()
            target = target.squeeze()
            # regular model
            with torch.no_grad():
                with autocast():
                    # regular model
                    regular_cls_output, regular_semantic_output = regular_model(inputData.cuda())
                    regular_cls_output = regular_cls_output.float()
                    regular_semantic_output = regular_semantic_output.float()

                    regular_overall_loss = criterion[0](regular_cls_output, target)
                    if use_cos:
                        regular_cos_loss = criterion[1](semantic.clone(), regular_semantic_output.cuda())*inputData.size(0)
                    else:
                        regular_cos_loss = torch.tensor(0.0)
                    # regular_cos_loss = torch.tensor(0.0)
                    regular_loss = regular_overall_loss + regular_cos_loss
            # ema model
            with torch.no_grad():
                ema_cls_output, ema_semantic_output = ema_model.module(inputData.cuda())
                ema_cls_output = ema_cls_output.float()
                ema_semantic_output = ema_semantic_output.float()
                ema_overall_loss = criterion[0](ema_cls_output, target)
                if use_cos:
                    ema_cos_loss = criterion[1](semantic.clone(), ema_semantic_output.cuda())*inputData.size(0)
                else:
                    ema_cos_loss = torch.tensor(0.0)
                ema_loss = ema_overall_loss + ema_cos_loss

            val_loader.set_postfix(cls=regular_overall_loss.item(), cos = regular_cos_loss.item(), e_cls=ema_overall_loss.item(), e_cos = ema_cos_loss.item())

            self.regular_ap_meter.add(regular_cls_output.data, target)
            self.ema_ap_meter.add(ema_cls_output.data, target)
            self.regular_loss_meter.add(regular_loss.item())
            self.ema_loss_meter.add(ema_loss.item())


    def train(self, models, train_loader, criterion, optimizer, scheduler, scaler, epoch):
        regular_model = models[0]
        ema_model = models[1]
        train_loader = tqdm(train_loader, desc='Train Epoch '+str(epoch))
        for i, (inputData, target, semantic) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            semantic = semantic[0].cuda().float()
            target = target.squeeze()
            with autocast():  # mixed precision
                cls_output, semantic_output = regular_model(inputData)
                cls_output = cls_output.float()
                semantic_output = semantic_output.float()
                
                cls_loss = criterion[0](cls_output, target)
                if use_cos:
                    cos_loss = criterion[1](semantic.clone(), semantic_output.cuda())*inputData.size(0)
                else:
                    cos_loss = torch.tensor(0.0)
            regular_loss = cls_loss + cos_loss

            regular_model.zero_grad()
            optimizer.param_groups[3]["lr"] = optimizer.param_groups[3]["lr"]*0.1
            optimizer.param_groups[2]["lr"] = optimizer.param_groups[2]["lr"]*0.1

            scaler.scale(regular_loss).backward()
            # loss.backward()
            scaler.step(optimizer)
            temp_lr = optimizer.param_groups[3]["lr"]
            scaler.update()
            # optimizer.step()
            scheduler.step()

            ema_model.update(regular_model)
            with torch.no_grad():
                # ema_cls_output, ema_semantic_output = ema_model.module(inputData, semantic)
                ema_cls_output, ema_semantic_output = ema_model.module(inputData.float())
                ema_cls_output = ema_cls_output.float()
                ema_semantic_output = ema_semantic_output.float()

            ema_cls_loss = criterion[0](ema_cls_output, target)
            if use_cos:
                ema_cos_loss = criterion[1](semantic.clone(), ema_semantic_output.cuda())*inputData.size(0)
            else:
                ema_cos_loss = torch.tensor(0.0)
            ema_loss = ema_cls_loss + ema_cos_loss
            
            # store information
            train_loader.set_postfix(cls=cls_loss.item(), cos = cos_loss.item(), e_cls=ema_cls_loss.item(), e_cos = ema_cos_loss.item(), lr=temp_lr)

            self.regular_ap_meter.add(cls_output.data, target)
            self.ema_ap_meter.add(ema_cls_output.data, target)
            self.regular_loss_meter.add(regular_loss.item())
            self.ema_loss_meter.add(ema_loss.item())


    def save_checkpoint(self, models, regular_ap, regular_map, ema_ap, ema_map, epoch):
        # save current model
        regular_model = models[0]
        ema_model = models[1]

        if not os.path.exists('LT_checkpoint/'):
            os.mkdir('LT_checkpoint/')

        # torch.save({
        #     'epoch': epoch,
        #     'score': regular_map,
        #     'model': 'regular',
        #     'state_dict': regular_model.state_dict(),
        # }, os.path.join('checkpoint/', 'regular_{}_cur.ckpt'.format(model_name)))

        # torch.save({
        #     'epoch': epoch,
        #     'score': ema_map,
        #     'model': 'ema',
        #     'state_dict': ema_model.module.state_dict(),
        # }, os.path.join('checkpoint/', 'EMA_{}_cur.ckpt'.format(model_name)))

        # # save best mAP model
        if regular_map > self.highest_regular_map:
            self.highest_regular_map = regular_map
        #     if self.highest_regular_model is not None:
        #         os.remove(os.path.join('checkpoint/', self.highest_regular_model))
        #     self.highest_regular_model = 'regular_{}_best_{score:.3f}_e{epo}.ckpt'.format(model_name, score=regular_map, epo=epoch)
        #     torch.save({
        #         'epoch': epoch,
        #         'score': regular_map,
        #         'model': 'regular',
        #         'state_dict': regular_model.state_dict(),
        #     }, os.path.join('checkpoint/', self.highest_regular_model))
            # if self.dataset == 'coco-lt' or self.dataset == 'voc-lt':
            #     ltAnalysis(regular_ap, self.dataset)


        if ema_map > self.highest_ema_map:
            self.highest_ema_map = ema_map
            if self.highest_ema_model is not None:
                os.remove(os.path.join('/data2/yanjiexuan/checkpoints/RC-Tran/LT_checkpoint/', self.highest_ema_model))
            self.highest_ema_model = 'ema_{}_best_{score:.3f}_e{epo}.ckpt'.format(model_name, score=ema_map, epo=epoch)
            torch.save({
                'epoch': epoch,
                'score': ema_map,
                'model': 'ema',
                'state_dict': ema_model.module.state_dict(),
            }, os.path.join('/data2/yanjiexuan/checkpoints/RC-Tran/LT_checkpoint/', self.highest_ema_model))
            if self.dataset == 'coco-lt' or self.dataset == 'voc-lt':
                ltAnalysis(ema_ap, self.dataset)
        
        # '_' means not the best
        self.highest_regular_model = 'regular_{}_best_e{epo}_{score:.3f}.ckpt'.format(model_name, epo=epoch, score=regular_map)
        cur = 'ema_{}_e{epo}_{score:.3f}.ckpt'.format(model_name, epo=epoch, score=ema_map)
        # if epoch in [9,10]:
        #     torch.save({
        #         'epoch': epoch,
        #         'score': regular_map,
        #         'model': 'regular',
        #         'state_dict': regular_model.state_dict(),
        #     }, os.path.join('checkpoint/', "_"+self.highest_regular_model))

        # if epoch in [22,23,24,25,26]:# for voc
        # if epoch in [11,12,13,14,15]: #for coco and nus-wide
        #     torch.save({
        #         'epoch': epoch,
        #         'score': ema_map,
        #         'model': 'ema',
        #         'state_dict': ema_model.module.state_dict(),
        #     }, os.path.join('/data2/yanjiexuan/checkpoints/RC-Tran/LT_checkpoint/', "_"+cur))    



        print('------------------------------------------------->>>>>>> Highest Experimental Results on {}'.format(model_name))
        print('Highest regular model mAP: {}\t Highest ema model mAP: {}'.format(self.highest_regular_map, self.highest_ema_map))