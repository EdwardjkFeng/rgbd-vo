#!/bin/bash
set -e 

evalset=(
	rgbd_bonn_balloon
	rgbd_bonn_balloon2
	rgbd_bonn_balloon_tracking
	rgbd_bonn_balloon_tracking2
	rgbd_bonn_crowd
	rgbd_bonn_crowd2
	rgbd_bonn_crowd3
	rgbd_bonn_kidnapping_box
	rgbd_bonn_kidnapping_box2
	rgbd_bonn_moving_nonobstructing_box
	rgbd_bonn_moving_nonobstructing_box2
	rgbd_bonn_moving_obstructing_box
	rgbd_bonn_moving_obstructing_box2
	rgbd_bonn_person_tracking
	rgbd_bonn_person_tracking2
	rgbd_bonn_placing_nonobstructing_box
	rgbd_bonn_placing_nonobstructing_box2
	rgbd_bonn_placing_nonobstructing_box3
	rgbd_bonn_placing_obstructing_box
	rgbd_bonn_removing_nonobstructing_box
	rgbd_bonn_removing_nonobstructing_box2
	rgbd_bonn_removing_obstructing_box
	rgbd_bonn_synchronous
	rgbd_bonn_synchronous2
	rgbd_bonn_static
	rgbd_bonn_static_close_far
)

TUM_DATASET="/home/jingkun/Dataset/Bonn_RGBD_Dataset"


# copy gt pose
# for seq in ${evalset[@]}; do
#   echo Copying gt pose from dataset
#   cp $TUM_DATASET/$seq/groundtruth.txt ./$seq/groundtruth.txt
#   # rm ./$seq/groundtruth.txt
# done


for seq in ${evalset[@]}; do
  echo $seq
  echo Evaluating APE ...
  cd ./$seq
	RESULT=ape_results
	mkdir -p $RESULT

	FILES=./*.txt
	SAVE_FORMAT=_ape.zip
	for f in $FILES
	do
	  echo Evaluating trajectory $f ...
	  new=${f/.txt/$SAVE_FORMAT}
	  evo_ape tum $TUM_DATASET/$seq/groundtruth.txt $f -as --save_results ${new/./$RESULT}
	done

	cd $RESULT
	evo_res ./*.zip --use_filenames --save_table table.csv


	cd ../
	echo Evaluating trans RPE ...

	RESULT=t_rpe_results
	mkdir -p $RESULT

	FILES=./*.txt
	SAVE_FORMAT=_rpe.zip
	for f in $FILES
	do
	  echo Evaluating trajectory $f ...
	  new=${f/.txt/$SAVE_FORMAT}
	  evo_rpe tum $TUM_DATASET/$seq/groundtruth.txt $f --pose_relation trans_part -as --delta 1 --delta_unit f --save_results ${new/./$RESULT}
	done

	cd $RESULT
	evo_res ./*.zip --use_filenames --save_table table.csv

  cd ../
	echo Evaluating rot RPE ...

	RESULT=r_rpe_results
	mkdir -p $RESULT

	FILES=./*.txt
	SAVE_FORMAT=_rpe.zip
	for f in $FILES
	do
	  echo Evaluating trajectory $f ...
	  new=${f/.txt/$SAVE_FORMAT}
	  evo_rpe tum $TUM_DATASET/$seq/groundtruth.txt $f --pose_relation angle_deg -as --delta 1 --delta_unit f --save_results ${new/./$RESULT}
	done

	cd $RESULT
	evo_res ./*.zip --use_filenames --save_table table.csv
  
  cd ../../
done