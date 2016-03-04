#! /usr/bin/env bash

repo="/home/leon/repos/saliency-localizer"
models="/home/leon/repos/saliency-localizer-models/"

$repo/scripts/find_tags.py \
    --weight-dir $models/season_2015 \
    --threshold 0.1 \
    --out /mnt/storage/beesbook/season_2015_preprocces/tags_positions.json \
    /mnt/storage/beesbook/season_2015_preprocces/images.txt
