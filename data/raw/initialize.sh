#!/bin/bash
corpus_root=/m/triton/scratch/cs/ussee/fi-parl/sessions
data_root=/m/triton/scratch/cs/ussee/fi-parl-facerec

for year in $corpus_root/20*;do
  for video in $year/session-*.mp4;do
    session_id=$(basename $video)
    session_id=${session_id%.mp4}
    session_id=${session_id#session-}
    mkdir $session_id
    cd $session_id
    ln -s $video $( basename $video )
    faces_id=$(echo $session_id | sed 's/-//')
    folder=$faces_id
    if [[ $folder == 00* ]]; then
        folder=$(echo $folder | sed 's/00//')
    elif [[ $folder == 0* ]]; then
        folder=$(echo $folder | sed 's/0//')
    fi
    
    scenes="$data_root/$folder-data/scene_changes"

    ln -s $scenes $(basename $scenes)
    features="$data_root/$folder-data/features"

    ln -s $features $(basename $features)
    faces="$data_root/$folder-data/session-$faces_id-faces.txt"
    ln -s $faces "session-$session_id-faces.txt"
    cd ..
  done
done
