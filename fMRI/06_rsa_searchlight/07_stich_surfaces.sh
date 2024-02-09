#!/bin/bash

pop="$1"
task="$2"

START=$(pwd)
OUT="$(mktemp -d /tmp/stich.XXXXX)"

cp "../bids_dataset/derivatives/rsa/sub-average/figures/pop-${pop}_task-${task}_"*.png "$OUT"

cd "$OUT" || exit 1

for f in *.png; do convert "$f" -transparent white tmp.png; mv tmp.png "$f"; done

for model in "IT" "symbolic" "both" "empty"; do

  convert -rotate -90 "pop-${pop}_task-${task}_view-ventral_hemisphere-left_model-${model}.png" b1.png
  convert -rotate -90 "pop-${pop}_task-${task}_view-ventral_hemisphere-right_model-${model}.png" b2.png

  convert -size 4000x4000 xc:transparent \
            -draw "image over 0070,0000 0,0 'pop-${pop}_task-${task}_view-lateral_hemisphere-left_model-${model}.png'" \
            -draw "image over 2040,0000 0,0 'pop-${pop}_task-${task}_view-lateral_hemisphere-right_model-${model}.png'" \
            -draw "image over 0820,0680 0,0 'b1.png'" \
            -draw "image over 1320,0680 0,0 'b2.png'" \
            -draw "image over 0000,0990 0,0 'pop-${pop}_task-${task}_view-medial_hemisphere-left_model-${model}.png'" \
            -draw "image over 2130,0990 0,0 'pop-${pop}_task-${task}_view-medial_hemisphere-right_model-${model}.png'" \
            "output.png"

  convert output.png -fuzz 1% -trim +repage "pop-${pop}_task-${task}_model-${model}_merged.png" 

  rm b1.png b2.png output.png

  cp "pop-${pop}_task-${task}_model-${model}_merged.png" "${START}/../bids_dataset/derivatives/rsa/sub-average/figures_stiched_tmp/"

done

rm -R "$OUT"
