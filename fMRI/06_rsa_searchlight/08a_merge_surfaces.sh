#!/bin/bash

pop="$1"
task="$2"

START=$(pwd)
OUT="$(mktemp -d /tmp/merge.XXXXX)"
cp "../bids_dataset/derivatives/rsa/sub-average/figures_stiched_tmp/pop-${pop}_task-${task}_model-"*.png $OUT

cd "$OUT"

to_merge=""

for model in "symbolic" "IT" "both"; do
  convert "pop-${pop}_task-${task}_model-${model}_merged.png" "pop-${pop}_task-${task}_model-empty_merged.png" \
    -alpha off +repage \( -clone 0 -clone 1 -compose difference -composite -threshold 0 \) -delete 1 -alpha off -compose copy_opacity -composite tmp.png
  #convert tmp.png -alpha set -background none -channel A -evaluate multiply 0.8 +channel "${model}_alone.png"
  cp tmp.png "${model}_alone.png"
  to_merge="$to_merge ${model}_alone.png"
done

convert "pop-${pop}_task-${task}_model-empty_merged.png" $to_merge -background none -layers flatten +repage "pop-${pop}_task-${task}_merged.png"

cp "pop-${pop}_task-${task}_merged.png" "${START}/../bids_dataset/derivatives/rsa/sub-average/figures_stiched/"

rm -R "$OUT"
