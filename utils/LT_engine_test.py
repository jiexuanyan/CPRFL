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

from randaugment import RandAugment
from config import *
import os

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

    def validate(self, model, val_loader):
        regular_model = model
        val_loader = tqdm(val_loader, desc='Test')
        for i, (inputData, target) in enumerate(val_loader):
            # compute output
            target = target.cuda()
            target = target.squeeze()
            # regular model
            with torch.no_grad():
                with autocast():
                    # regular model
                    regular_cls_output, _ = regular_model(inputData.cuda())
                    regular_cls_output = regular_cls_output.float()

            self.regular_ap_meter.add(regular_cls_output.data, target)

