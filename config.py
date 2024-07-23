# # experimental setup

# # model
# head_num = 4
# depth = 1
# feature_d = 1024
# dim_head = 512
# # ff_neck 1024 for 1x
# mlp_dim = 1024

# # sample_rate = 1
# fpn = False
# label_emb_drop = False
# # pos_emb = False
# # sample_num = 1
# # use_full_fm = True
# # glo_channel_concat = True
# # engine
# use_cos = False

# nonlinear = True
# out2_neck = True

# #
# model_name = 'Encoder_nonlinear_adj_2out_5th'

# experimental setup

# model
head_num = 8

depth = 1
feature_d = 2048
# feature_d = 1024
dim_head = 512
# ff_neck
mlp_dim = 2048

# label_emb是否drop_out,默认False,True性能会下降
label_emb_drop = False
# decoder
# fpn = False
# sample_rate = 1
# pos_emb = False
# sample_num = 1
# use_full_fm = True

glo_channel_concat = False

# engine
use_cos = False

nonlinear = True
# nonlinear = False
out2_neck = False
expand = 0.5


no_cls_grad = False
# no_direct_grad=False
no_feature_grad = False


linear_name = "nonlinear" if nonlinear else "linear"

#

model_name = 'Encoder_{}_d{}_{}_mlp{}_{}x_lr5e-5_asl_clip_coco_fine_visual'.format(linear_name,head_num,feature_d,mlp_dim,expand)



# # C-Tran settings
# num_classes =6
# use_lmt=False
# pos_emb=False
# layers=3
# heads=4
# dropout=0.1
# int_loss=0
# no_x_features=False

# # model
# head_num = 8
# depth = 1
# feature_d = 2048
# dim_head = 512
# # ff_neck
# mlp_dim = 4096

# # sample_rate = 1
# fpn = False
# label_emb_drop = False
# # pos_emb = False
# # sample_num = 1
# # use_full_fm = True
# # glo_channel_concat = True
# # engine
# use_cos = False

# nonlinear = True
# out2_neck = True

# #
# model_name = 'Encoder_nonlinear_d8_2048_mlp4096_adj_2out'



#------------legacy 3.4 mAP70.02-----------------
## experimental setup 

# # model
# head_num = 8
# depth = 1
# feature_d = 2048
# dim_head = 512
# # ff_neck
# mlp_dim = 2048

# # sample_rate = 1
# fpn = False
# label_emb_drop = False
# # pos_emb = False
# # sample_num = 1
# # use_full_fm = True
# # glo_channel_concat = True
# # engine
# use_cos = False

# nonlinear = True
# out2_neck = False

# #
# model_name = 'Encoder_nonlinear_d8_2048_mlp2048'