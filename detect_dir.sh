#!/bin/bash
# run python detect.py on all video files in a specified directory

directory=$1

echo "Processing all video files in $directory"
for video in $(find "$directory" -type f -name "*.mp4"); do
    echo "Processing $video"
    python3 detect.py --weights weights/pruned_model_plus_20_v1.pt --source "$video" --hide-conf --hide-labels --line-thickness=1 --project runs/detect/test_all_studata
done
    