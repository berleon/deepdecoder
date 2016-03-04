#! /usr/bin/env bash

images="/mnt/storage/beesbook/season_2015_preprocces/images.txt"
json="/home/leon/data/tags_positions.json"
tmp_dir="/tmp/beesbook_links/"
source_dir="/mnt/storage/beesbook/season_2015_images/"
repo="/home/leon/repos/saliency-localizer"

if [ "$1" == "init_links" ]; then
    mkdir -p $tmp_dir

    for image in $(cat $images); do
        name=$(basename $image)
        ln -sf $source_dir/${name/_wb.jpeg/.jpeg} $tmp_dir/$name
    done
fi


$repo/scripts/build_hdf5_dataset.py  \
    --out /home/leon/data/tags_plain_t6.hdf5 \
    --roi-size 64 \
    --offset -36 \
    --threshold 0.6 \
    --image-dir $tmp_dir \
    $json