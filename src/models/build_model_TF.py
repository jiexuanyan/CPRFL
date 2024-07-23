from .avg_pool import FastAvgPool2d
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from .TF import Transformer
from exp import *

import clip

# from .clip_text import CLIP_Text

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

class MyModel(nn.Module):

    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device = torch.device('cuda' if torch.cuda.is_available() else "cpu"))
        return prompts
    
    def __init__(self, args,classnames,clip_model):
        super(MyModel, self).__init__()
        # create backbone model
        model = models.resnet101(pretrained=False)
        if args['pretrained']:
            model.load_state_dict(torch.load(args['pretrained']))
        else:
            print('cannot find the pre-trained backbone model in this path : {}'.format(args['pretrained']))

        # keep the convolutional layers as backbone
        self.Backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        self.attention = Transformer(
            dim=feature_d,
            depth=depth,
            heads=head_num,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=args['drop']
        )

        print('dropout rate: ', args['drop'])

        self.prompts = self.get_tokenized_prompts(classnames)
        self.text_encoder = TextEncoder(clip_model)
        # print("fc gelu")
        # not work here
        # self.fc = nn.Sequential(
        #     nn.Linear(2048, feature_d*2),
        #     nn.Identity(),
        #     nn.GELU(),
        #     # nn.ReLU(),
        #     nn.Linear(feature_d*2, feature_d)
        #     )
        
        if not args['train']:
            print("==============> Eval with visual dropout.")
            self.fc = nn.Sequential(
                nn.Linear(2048, feature_d),
                nn.Dropout(p=args['drop'])
                )
        else:
            self.fc = nn.Sequential(
                nn.Linear(2048, feature_d),
                # nn.Dropout(p=0.3)
                )

        if not nonlinear:
            if label_emb_drop:  
                self.label_emb = nn.Sequential(
                    nn.Linear(1024, feature_d),
                    nn.Dropout(p=args['drop'])
                )
            else:
                self.label_emb = nn.Linear(1024, feature_d)
        else:
            if label_emb_drop: 
                print("==============> Eval with emb dropout.") 
                self.label_emb = nn.Sequential(
                    nn.Linear(1024, feature_d*2),
                    nn.GELU(),
                    nn.Dropout(args['drop']),
                    nn.Linear(feature_d*2, feature_d),
                    nn.Dropout(args['drop'])
                )
            else:
                self.label_emb = nn.Sequential(
                    nn.Linear(1024, int(feature_d*expand)),
                    nn.Identity(),
                    nn.GELU(),
                    # nn.ReLU(),
                    nn.Linear(int(feature_d*expand), feature_d)
                )

        # pooling layer
        self.pooling = FastAvgPool2d(flatten=True)
        # fully connected layer configuration
        self.dim_features = 2048
        #CLIP_RN50的输出特征为1024，CLIP_ViT16的输出特征为512
        self.dim_semantic = 1024

    # def encode_text(self, text):
    #     try:
    #         text_features = self.text_encoder(text)
    #     except:
    #         # CUDA out of memory
    #         text_split = torch.split(text, 1000)
    #         text_features = torch.cat([self.text_encoder(x) for x in text_split])
    #     return text_features
        
    def forward(self, img):
    # def forward(self,img):
        batch_size = img.size(0)

        prompts = self.prompts
        text_features = self.text_encoder(prompts)

        text_features = text_features.float()
        attr = self.label_emb(text_features).unsqueeze(0).expand(batch_size, text_features.size(0), feature_d)
        # attr = text_features.unsqueeze(0).expand(batch_size, text_features.size(0), feature_d)
 
        feature_maps = self.Backbone(img)
  
        feature_maps = feature_maps.view(feature_maps.size(0), feature_maps.size(1), -1).clone().transpose(-1, -2)
        feature_maps = self.fc(feature_maps)
        
        feature = torch.cat((feature_maps, attr), 1)

        feature = self.attention(feature)
        # 阻断编码器回传梯度
        if no_feature_grad:
            feature_ = feature.detach()
            classify_feature = feature_[:, -attr.size(1):, :]
        else:
            classify_feature = feature[:, -attr.size(1):, :]

        classify_feature = classify_feature.unsqueeze(-2)
        attr = attr[0].unsqueeze(-1)
        

        # 阻断类语义原型回流梯度
        if no_cls_grad:
            attr_ = attr.detach()
            score = torch.matmul(classify_feature, attr_).squeeze()
        else:
            score = torch.matmul(classify_feature, attr).squeeze()

        # score = torch.matmul(classify_feature, attr).squeeze()

        return score, attr.squeeze()
