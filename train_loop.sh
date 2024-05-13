#!/bin/bash

# Set the base directory
base_dir="/home/CoAtNet/pytorch-image-models/output/train"

while true; do

    # Get the list of directories in base_dir sorted by the most recent modification time
    directories=$(ls -d $base_dir/* | sort -r)

    # Iterate through the directories in order
    for dir in $directories; do
        # Check if the summary.csv file exists in the directory
        if [ -f "$dir/summary.csv" ]; then
            # Read the last line of summary.csv to get the latest epoch
            latest_epoch=$(tail -n 1 "$dir/summary.csv" | cut -d ',' -f1)

            # print the latest epoch
            echo "Latest epoch in directory $dir: $latest_epoch"

            # Check if the latest epoch is less than 29
            if [ "$latest_epoch" -lt 199 ]; then
                # Set the args.yaml file path
                config_file="$dir/args.yaml"

                # Set the last.pth.tar file path
                checkpoint_file="$dir/last.pth.tar"

                echo "Resuming training in directory $dir with config file: $config_file and checkpoint file: $checkpoint_file"

                # Run the train command with the appropriate config and resume options
                python train.py --config "$config_file" --resume "$checkpoint_file"

                # After resuming, read the last epoch again
                directories=$(ls -d $base_dir/* | sort -r)
                check_directory="${directories[0]}"

                if [ -f "$check_directory/summary.csv" ]; then
                    latest_epoch=$(tail -n 1 "$check_directory/summary.csv" | cut -d ',' -f1)

                    # If latest_epoch is now 199, exit the loop and stop the script                
                    if [ "$latest_epoch" -eq 199 ]; then
                        echo "Training has completed up to epoch 29 in directory $check_directory. Exiting script."
                        exit 0
                    fi
                fi

                # Exit the loop once the training is resumed
                break
            fi
        fi
    done

    echo "sleep 10 sec before checking again."
    sleep 10
done