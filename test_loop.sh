#!/bin/bash

# Set the base directory
base_dir="/home/CoAtNet/pytorch-image-models/output/train"

# Get the list of directories in base_dir sorted by the most recent modification time
directories=$(ls -d $base_dir/* | sort -r)

# Define an array of dataset information as strings
datasets=(
    "/usr/src/data/20240325_gen_xlarge test"
    "/usr/src/data/20240326_mapillary_no_resize val"
    "/usr/src/data/20240319_tsinghua_cleaned test"
    "/usr/src/data/20240319_hyundai_partridge_systems_test/14626_no_resize test"
)

# Iterate through the directories in order
for dir in $directories; do
    # Check if the summary.csv file exists in the directory
    if [ -f "$dir/summary.csv" ]; then
        # Set the last.pth.tar file path
        checkpoint_file="/home/CoAtNet/pytorch-image-models/output/train/20240429-041632-coatnet_0_rw_224_sw_in1k/checkpoint-104.pth.tar"

        for dataset_info in "${datasets[@]}"; do
            # Split the string into root_path and split_name using read with IFS
            IFS=' ' read -r root_path split_name <<< "$dataset_info"
            
            # Use the root_path and split_name as needed
            echo "Root dataset path: $root_path"
            echo "Split folder name: $split_name"
            
            echo "Test dataset at $root_path with split $split_name..."

            # test the checkpoint model on test dataset
            python validate.py --data-dir $root_path --split $split_name --model coatnet_0_rw_224.sw_in1k --num-classes 2 --checkpoint "$checkpoint_file"
            
            echo "Finished test dataset at $root_path with split $split_name (checkpoint file: $checkpoint_file)."
        done

        exit 0

        # Exit the loop once the training is resumed
        break
    fi
done