#!/bin/bash
corpus_root=/m/triton/scratch/cs/ussee/fi-parl/sessions
data_root=/m/scratch/cs/ussee/fi-parl-rec

for year in $corpus_root/20*;do
  for video in $year/session-*.mp4;do
    session_id=$(basename $video)
    session_id=${session_id%.mp4}
    session_id=${session_id#session-}
    mkdir $session_id
    cd $session_id
    ln -s $video $( basename $video )
    faces=$session_id
    if [[ $faces == 00* ]]; then
        faces=$(echo $faces | sed 's/00//')
    elif [[ $faces == 0* ]]; then
        faces=$(echo $faces | sed 's/0//')
    fi
    faces=$(echo $faces | sed 's/-//')
    scenes="$data_root/$faces-data/scene_changes"
    ln -s $scenes scene_changes
    features="$data_root/$faces-data/features"
    ln -s $features features
    cd ..
  done
done