import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from utils.util import *
from utils.util import AveragePrecisionMeter
from torch.cuda.amp import GradScaler, autocast
import torchnet as tnt
from torch.optim import lr_scheduler
from src.data_loader.coco_fsl import CocoDatasetAugmentation
from src.data_loader.nus_fsl import NUSWIDEClassification_fsl
from src.data_loader.voc_fsl import Voc2007Classification_fsl
from randaugment import RandAugment
from config import *
import os
use_cos = True

class MultiLabelEngine():
    def __init__(self, args):
        # hyper-parameters
        self.evaluation = args['evaluation']
        self.thre = args['threshold']
        self.threshold = args['threshold']
        print('-----------test start: ')

        # measure mAP
        print("thre:", self.thre)
        self.regular_ap_meter = AveragePrecisionMeter(threshold=self.threshold, difficult_examples=False)

        

    def meter_reset(self):
        self.regular_ap_meter.reset()

    # def meter_print_val(self):
    #     print("starting metric r......")
    #     regular_ap = 100 * self.regular_ap_meter.value()
    #     regular_map = regular_ap.mean()

    #     print('=================================================>>>>>>> Experimental Results')
    #     print('regular mAP score: {map:.3f}'.format(map=regular_map))


    #     OP, OR, OF1, CP, CR, CF1 = self.regular_ap_meter.overall()
    #     OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.regular_ap_meter.overall_topk(3)
    #     print('CP: {CP:.4f}\t'
    #           'CR: {CR:.4f}\t'
    #           'CF1: {CF1:.4f}'
    #           'OP: {OP:.4f}\t'
    #           'OR: {OR:.4f}\t'
    #           'OF1: {OF1:.4f}\t'.format(CP=CP, CR=CR,
    #                                   CF1=CF1, OP=OP, OR=OR, OF1=OF1))
    #     print('OP_3: {OP:.4f}\t'
    #           'OR_3: {OR:.4f}\t'
    #           'OF1_3: {OF1:.4f}\t'
    #           'CP_3: {CP:.4f}\t'
    #           'CR_3: {CR:.4f}\t'
    #           'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k,
    #                                     CR=CR_k, CF1=CF1_k))
        

    #     return regular_map


    def meter_print(self):
        regular_ap = 100 * self.regular_ap_meter.value()
        regular_map = regular_ap.mean()
        reg_meters = self.regular_ap_meter.overall()
        OP, OR, OF1, CP, CR, CF1 = reg_meters
        reg_topk = self.regular_ap_meter.overall_topk(3)
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = reg_topk
        print('=================================================>>>>>>> Experimental Results on regular {}'.format(model_name))
        print('mAP score: {map:.3f}\t'.format(map=regular_map))
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

        return regular_ap, regular_map, reg_meters, reg_topk
    
    # def set_layer(self,layer:nn.Module,freeze = True):
    #     if freeze:
    #         for param in layer.parameters():
    #             param.requires_grad = False

    #     else:
    #         for param in layer.parameters():
    #             param.requires_grad = True

    def learning(self, model, test_loader):
        # regular_aps = {}
        # regular_map = 0
        # reg_meter_ls = []
        # reg_topk_ls = []
        # filtered_dict['fc.3.weight'] = filtered_dict.pop('fc.2.weight')
        # filtered_dict['fc.3.bias'] = filtered_dict.pop('fc.2.bias')
        try:
            if self.evaluation:
                model.eval()
                self.meter_reset()
                self.validate(model, test_loader)
                regular_ap, regular_map, reg_meters, reg_topk = self.meter_print()
                print(regular_ap.type)
                print('======================>>>>>>> test Experimental Results <<<<<<<======================')
                print(' regular {} model mAP :{}'.format(model_name, regular_map))
                # regular_aps.append(regular_ap)
                # # regular_maps.append(regular_map)
                # reg_meter_ls.append(reg_meters)
                # reg_topk_ls.append(reg_topk)

        except Exception as e:
            print("Error happened!") 
            print(e)                
        return regular_ap,regular_map, reg_meters, reg_topk

    # def train(self, model, train_loader, criterion, optimizer, scheduler, scaler, epoch, epi):
    #     regular_model = model
    #     train_loader = tqdm(train_loader, desc='Train Epi {} Epoch {}'.format(epi, epoch))
    #     for i, (inputData, target, semantic) in enumerate(train_loader):
    #         inputData = inputData.cuda()
    #         target = target.cuda()
    #         semantic = semantic[0].cuda().float()
    #         with autocast():  # mixed precision
    #             cls_output, semantic_output = regular_model(inputData, semantic)
    #             cls_output = cls_output.float()
    #             semantic_output = semantic_output.float()

    #         cls_loss = criterion[0](cls_output, target)
    #         if use_cos:
    #             cos_loss = criterion[1](semantic.clone(), semantic_output.cuda())*inputData.size(0)
    #         else:
    #             cos_loss = torch.tensor(0.0)
    #         regular_loss = cls_loss + self.alpha*cos_loss

            # if epoch%10 < 5 :
            #     self.set_layer(regular_model.attention,freeze = True)
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            #     self.set_layer(regular_model.attention,freeze = False)

            # elif epoch%10 >= 5 and epoch%10 < 9:
            #     self.set_layer(regular_model.label_emb,freeze = True)
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            #     self.set_layer(regular_model.label_emb,freeze = False)

            # else:
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()

            
            # if epoch%10 < 5 :
            #     self.set_layer(regular_model.label_emb,freeze = True)
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            #     self.set_layer(regular_model.label_emb,freeze = False)

            # elif epoch%10 >= 5 and epoch%10 < 9:
            #     self.set_layer(regular_model.attention,freeze = True)
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            #     self.set_layer(regular_model.attention,freeze = False)

            # else:
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            # regular_model.zero_grad()
            # scaler.scale(regular_loss).backward()
            # # loss.backward()
            # scaler.step(optimizer)
            # scaler.update()
            # # optimizer.step()
            # scheduler.step()

            # # store information
            # train_loader.set_postfix(cls=cls_loss.item(), cos = cos_loss.item())
            # self.regular_ap_meter.add(cls_output.data, target)
            # self.regular_loss_meter.add(regular_loss.item())

    def validate(self, model, val_loader):
        regular_model = model
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

            self.regular_ap_meter.add(regular_cls_output.data, target)

