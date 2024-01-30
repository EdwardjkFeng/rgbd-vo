#!/bin/bash
set -e 

evalset=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_xyz
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg2_desk
    rgbd_dataset_freiburg2_360_hemisphere
    rgbd_dataset_freiburg2_large_no_loop
    rgbd_dataset_freiburg2_large_with_loop
    rgbd_dataset_freiburg2_pioneer_360
    rgbd_dataset_freiburg2_pioneer_slam
    rgbd_dataset_freiburg2_pioneer_slam2
    rgbd_dataset_freiburg2_pioneer_slam3
    rgbd_dataset_freiburg2_xyz
    rgbd_dataset_freiburg2_rpy
    rgbd_dataset_freiburg2_dishes
    rgbd_dataset_freiburg2_360_kidnap
    rgbd_dataset_freiburg2_desk_with_person
    rgbd_dataset_freiburg3_teddy
    rgbd_dataset_freiburg3_long_office_household
    rgbd_dataset_freiburg3_sitting_static
    rgbd_dataset_freiburg3_sitting_rpy
    rgbd_dataset_freiburg3_sitting_xyz
    rgbd_dataset_freiburg3_sitting_halfsphere
    rgbd_dataset_freiburg3_walking_static
    rgbd_dataset_freiburg3_walking_xyz
    rgbd_dataset_freiburg3_walking_rpy
    rgbd_dataset_freiburg3_walking_halfsphere
)

TUM_DATASET="/home/jingkun/Dataset/TUM_RGBD_Dataset"


# copy gt pose
# for seq in ${evalset[@]}; do
#   echo "Copying gt pose from dataset"
#   cp $TUM_DATASET/$seq/groundtruth.txt ./$seq/groundtruth.txt
#   # rm ./$seq/groundtruth.txt
# done


for seq in ${evalset[@]}; do
  echo "$seq"
  echo "Evaluating APE ..."
  cd ./$seq
	RESULT="ape_results"
	mkdir -p $RESULT

	FILES="./*.txt"
	SAVE_FORMAT="_ape.zip"
	for f in $FILES
	do
	  echo "Evaluating trajectory $f ..."
	  new=${f/.txt/$SAVE_FORMAT}
	  evo_ape tum $TUM_DATASET/$seq/groundtruth.txt $f -as --save_results ${new/./"$RESULT"}
	done

	cd $RESULT
	evo_res ./*.zip --use_filenames --save_table table.csv


	cd ../
	echo "Evaluating trans RPE ..."

	RESULT="t_rpe_results"
	mkdir -p $RESULT

	FILES="./*.txt"
	SAVE_FORMAT="_rpe.zip"
	for f in $FILES
	do
	  echo "Evaluating trajectory $f ..."
	  new=${f/.txt/$SAVE_FORMAT}
	  evo_rpe tum $TUM_DATASET/$seq/groundtruth.txt $f --pose_relation trans_part -as --delta 1 --delta_unit f --save_results ${new/./"$RESULT"}
	done

	cd $RESULT
	evo_res ./*.zip --use_filenames --save_table table.csv

  cd ../
	echo "Evaluating rot RPE ..."

	RESULT="r_rpe_results"
	mkdir -p $RESULT

	FILES="./*.txt"
	SAVE_FORMAT="_rpe.zip"
	for f in $FILES
	do
	  echo "Evaluating trajectory $f ..."
	  new=${f/.txt/$SAVE_FORMAT}
	  evo_rpe tum $TUM_DATASET/$seq/groundtruth.txt $f --pose_relation angle_deg -as --delta 1 --delta_unit f --save_results ${new/./"$RESULT"}
	done

	cd $RESULT
	evo_res ./*.zip --use_filenames --save_table table.csv
  
  cd ../../
done