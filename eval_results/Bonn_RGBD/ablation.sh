#!/bin/bash
set -e 

evalset=(
	rgbd_bonn_balloon2
	rgbd_bonn_balloon_tracking2
	rgbd_bonn_crowd3
	rgbd_bonn_person_tracking2
	rgbd_bonn_placing_nonobstructing_box
)

TUM_DATASET="/home/jingkun/Dataset/Bonn_RGBD_Dataset"

methods=(
	# ours_F+P
	# ours_F+U
	# ours
	# ours_US
	ours_F
)


for seq in ${evalset[@]}; do
	echo $seq
	echo Evaluating APE ...
	cd ./$seq
	for m in ${methods[@]}; do
		f=./$m.txt
		echo Evaluating trajectory $f ...
		evo_ape tum $TUM_DATASET/$seq/groundtruth.txt $f -as
	done

	echo Evaluating trans RPE ...

	for m in ${methods[@]}; do
		f=./$m.txt
		echo Evaluating trajectory $f ...
		evo_rpe tum $TUM_DATASET/$seq/groundtruth.txt $f --pose_relation trans_part -as --delta 1 --delta_unit f
  	done

	echo Evaluating rot RPE ...
	
	for m in ${methods[@]}; do
		f=./$m.txt
		echo Evaluating trajectory $f ...
		evo_rpe tum $TUM_DATASET/$seq/groundtruth.txt $f --pose_relation angle_deg -as --delta 1 --delta_unit f
  	done
	cd ../
done
