# # experimental setup

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

glo_channel_concat = False


nonlinear = True
# nonlinear = False
out2_neck = False
expand = 0.5


no_cls_grad = False
# no_direct_grad=False
no_feature_grad = False


linear_name = "nonlinear" if nonlinear else "linear"

model_name = 'Encoder_{}_d{}_{}_mlp{}_{}x_lr5e-5_asl_clip_coco_fine_visual'.format(linear_name,head_num,feature_d,mlp_dim,expand)



