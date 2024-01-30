#!/bin/bash
set -e 

evalset=(
	rgbd_dataset_freiburg1_360
	rgbd_dataset_freiburg1_desk
	rgbd_dataset_freiburg2_desk
	rgbd_dataset_freiburg2_pioneer_360
	rgbd_dataset_freiburg3_walking_static
	rgbd_dataset_freiburg3_walking_xyz
)

TUM_DATASET="/home/jingkun/Dataset/TUM_RGBD_Dataset"

methods=(
	# ours_F+P
	# ours_F+U
	# ours_GS
	ours_F
	# ours
)


for seq in ${evalset[@]}; do
	echo $seq
	echo Evaluating APE ...
	cd ./$seq
	for m in ${methods[@]}; do
		f=./$m.txt
		echo Evaluating trajectory $f ...
		evo_ape tum $TUM_DATASET/$seq/groundtruth.txt $f -as || continue
	done

	echo Evaluating trans RPE ...

	for m in ${methods[@]}; do
		f=./$m.txt
		echo Evaluating trajectory $f ...
		evo_rpe tum $TUM_DATASET/$seq/groundtruth.txt $f --pose_relation trans_part -as --delta 1 --delta_unit f || continue
  	done

	echo Evaluating rot RPE ...
	
	for m in ${methods[@]}; do
		f=./$m.txt
		echo Evaluating trajectory $f ...
		evo_rpe tum $TUM_DATASET/$seq/groundtruth.txt $f --pose_relation angle_deg -as --delta 1 --delta_unit f || continue
  	done
	cd ../
done
