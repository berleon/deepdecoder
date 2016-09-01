#! /usr/bin/env bash

video_dir=$1
name=$2
nb_videos=5000

find $video_dir -name '*.mkv' | sort -R | head -n $nb_videos >  $name
