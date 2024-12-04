#!/bin/bash
# run python detect.py on all video files in a specified directory

directory=$1
conf_thres=$2
echo "Processing all video files in $directory"
for video in $(find "$directory" -type f -name "*.mp4"); do
    echo "Processing $video"
    python3 detect.py --weights weights/pruned_model_plus_20_v1.pt --source "$video" --hide-conf --hide-labels --line-thickness=1 --project runs/detect_all_40 --conf-thres $2
done
