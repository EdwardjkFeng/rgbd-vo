#!/bin/bash

python -B experiment/rgbd_vo.py \
--dataset TUM_RGBD \
--encoder_name RGB \
--mestimator None \
--solver Direct-Nodamping \
--cpu_workers 12 \
--batch_per_gpu 96 \
--feature_channel 8 \
--feature_extract conv \
--uncertainty laplacian \
--uncertainty_channel 1 \
--direction inverse \
--init_pose sfm_net \
--train_init_pose \
--multi_hypo prob_fuse \
--remove_tru_sigma \
--checkpoint checkpoint/dynamic_finetune_checkpoint_epoch39.pth.tar \
--vo RGB \
--image_resize 1 \
--time \
--vis_feat \

# --combine_ICP \

# --max_iter_per_pyr 20

# keyframe tracking visualization
# --vo_type keyframe --two_view