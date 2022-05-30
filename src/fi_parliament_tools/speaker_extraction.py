"""Command line client for face extraction Finnish Parliament data."""
from fi_parliament_tools.pipeline import Pipeline
from typing import List
from typing import Dict
from typing import Any
from pathlib import Path
import json
from itertools import combinations
from collections import defaultdict
from typing import DefaultDict
import numpy as np
from alive_progress import alive_bar
from syncnet.run_syncnet import run_syncnet
from syncnet.run_pipeline import run_pipeline
from argparse import Namespace
import shutil
from fi_parliament_tools.video_utils.create_videos import VideoCreatorPipeline
from fi_parliament_tools.audio_alignment import AudioAlignmentPipeline
from logging import Logger
import time

class SpeakerExtractionPipeline(Pipeline):
    def __init__(
        self, log: Logger, data_path: str, aligned_path: str, session_name: str
    ) -> None:
        super().__init__(log)
        self.data_path = data_path
        self.session_name = session_name
        self.metadata_path = Path(data_path, session_name, f"{session_name}_data.json")
        self.video_path = Path(data_path, session_name, f"session-{session_name}.mp4")
        self.aligned_path = Path(aligned_path, self.session_name)

        self.distance_threshold = 200
        self.missing_threshold = 100
        self.filter_threshold = 400
        self.cost_threshold = 40
        self.frames_per_clip = 200

        self.width_height = [1920, 1080]
        self.center_of_vid = [elem / 2 for elem in self.width_height]

        self.tmp_path = Path(self.data_path, self.session_name, "scene_analysis")
        self.tmp_path.mkdir(exist_ok=True)
        
    def load_metadata(self, path: Path):
        if path.exists():
            with path.open(mode="r", encoding="utf-8", newline="") as file:
                video_metadata = json.load(file)
                scenes = video_metadata["scenes"]
                frames = video_metadata["frames"]
                frames = {int(frame): values for frame, values in frames.items()}
            return frames, scenes
        else:
            raise OSError(f"Could not open/read file: {path}")

    def obtain_center_bb(self, coords): 
        center_point = [coords[0] / 2 + coords[2], coords[1] / 2 + coords[3]]
        return center_point

    def find_centered_speakers(self, frame_data: List[Dict[str, Any]]):
        frame_coords = [speaker_info["coords"] for speaker_info in frame_data]
        speaker_names = [speaker_info["speaker_id"] for speaker_info in frame_data]
        conv_coords = [self.obtain_center_bb(coord) for coord in frame_coords]

        diff = np.array(conv_coords) - np.array(self.center_of_vid)
        distances = np.linalg.norm(diff, axis=1)  # type: ignore
        indices = np.argsort(distances)
        condition = (distances <= self.distance_threshold + distances[indices[:1]])
        centered_inds = np.where(condition)[0]
        speakers = [frame_data[index] for index in centered_inds]
        return speakers

    def rename_duplicated_speakers(self, speakers): 
        # There are some frames in which the speakers are duplicated, meaning that there are the same ids in the same frame.
        ids = [data['speaker_id'] for data in speakers]
        duplicated_names = defaultdict(list)
        for i, item in enumerate(ids):
            duplicated_names[item].append(i)
        for _, locs in duplicated_names.items():
            for i in range(1, len(locs)):
                speakers[locs[i]]['speaker_id'] = speakers[locs[i]]['speaker_id']+'d'+str(i)
        ids = [data['speaker_id'] for data in speakers]   
        return speakers

    def compute_distances(self, speakers, faces):
        faces_coords = np.array([self.obtain_center_bb(data[-1][1]) for data in faces.values()])
        faces_names = np.array([data for data in faces.keys()])
        speaker_coords = np.array([self.obtain_center_bb(speaker['coords']) for speaker in speakers])
        distances = np.zeros((len(speaker_coords), len(faces_coords)))
        if len(faces_coords)!=0: # In other words, if its not the first frame
            for i, speaker in enumerate(speaker_coords):
                for j, face in enumerate(faces_coords):
                    distances[i][j] = np.linalg.norm(face - speaker)
        return distances 

    def determine_speaker_position(self, distances, speakers):
        # https://stackoverflow.com/questions/31694080/get-an-index-of-a-sorted-matrix
        sorted_min_values = list(zip(*np.argsort(distances, axis=None).__divmod__(distances.shape[1])))
        positions = [-1] * len(speakers)
        speakers_used = []
        indices_used = []
        for speaker, index in sorted_min_values:
            cost = distances[speaker][index]
            if speaker not in speakers_used and index not in indices_used:
                    if cost <= self.cost_threshold: 
                        positions[speaker] = index
                        indices_used.append(index)
                        speakers_used.append(speaker)
        return positions

    def detect_faces_in_scene(
        self,
        frames: DefaultDict[int, List[Dict[str, Any]]],
        start_scene: int,
        end_scene: int,
    ) -> None:
        saved_faces = defaultdict(list)
        for frame in range(start_scene, end_scene):
            if frame in frames.keys():
                frame_data = frames[frame]
                speakers = self.find_centered_speakers(frame_data)
                speakers = self.rename_duplicated_speakers(speakers)
                distances = self.compute_distances(speakers, saved_faces)
                positions = self.determine_speaker_position(distances, speakers)  
                for j, speaker in enumerate(speakers):
                    name = speaker["speaker_id"]
                    frame_coords = (frame, speaker["coords"])
                    if positions[j]==-1:
                        name += '_x' if name in saved_faces.keys() else ''
                    else:
                        name = list(saved_faces.keys())[positions[j]]
                    saved_faces[name].append(frame_coords)
        return saved_faces

    def filter_short_videos(self, faces): 
        new_faces = defaultdict(list)
        for face in faces.items():
            speaker, coords_list = face
            if len(coords_list) >= self.filter_threshold:
                new_faces[speaker] = coords_list
        return new_faces

    def fix_missing_frames(self, frames): 
        fixed_frames = defaultdict(list)
        for speaker, frame_info in frames.items():
            name = speaker
            count = 0
            last_frame = frame_info[0][0]
            fixed_frames[name].append(frame_info[0])
            for next_frame, next_coords in frame_info[1:]:
                if (next_frame - last_frame) < self.missing_threshold:
                    for new_frame in range(last_frame+1, next_frame):
                        fixed_frames[name].append((new_frame, next_coords))
                elif next_frame-1 != last_frame:
                    self.log.info(f"Missing frames between {last_frame}-{next_frame-1}")
                    count += 1
                    name = speaker + '_' + str(count)
                fixed_frames[name].append((next_frame, next_coords))
                last_frame = next_frame 
        return fixed_frames

    def run_syncnet(self, video):
        model_path = Path("models", "syncnet_v2.model")
        ref_name = video.split("/")[-2:]
        ref_name = "/".join(ref_name)[:-4]
        tmp_path = Path(self.data_path, "syncnet_tmp")
        opt = Namespace(
            initial_model=model_path,
            batch_size=20,
            vshift=15,
            videofile=video,
            data_dir=tmp_path,
            reference=ref_name,
        )
        run_pipeline(opt)
        _, conf, _ = run_syncnet(opt)
        shutil.rmtree(tmp_path)
        return float(conf)

    def divide_speeches_in_clips(self, coords): 
        last_frame, first_frame = int(coords[-1][0]), int(coords[0][0])
        total_frames = last_frame - first_frame
        n_clips, remainder = divmod(total_frames, self.frames_per_clip)
        frames_of_speaker = []
        coords_of_speaker = []
        coords_length = 0
        for i in range(0, int(n_clips)):
            start_frame = first_frame + i * self.frames_per_clip
            if i == int(n_clips) - 1:
                end_frame = start_frame + self.frames_per_clip + int(remainder)
            else:
                end_frame = start_frame + self.frames_per_clip
            frames_of_speaker.append((start_frame, end_frame))
            length_frames = end_frame - start_frame
            coords_to_use = coords[coords_length : coords_length + length_frames]
            coords_to_use = [coords[1] for coords in coords_to_use]
            coords_length += length_frames
            coords_of_speaker.append(coords_to_use)
        return frames_of_speaker, coords_of_speaker

    def detect_lip_activity(self, videos, n_scene, speaker_frames, speaker_coords):
        lip_activity = []
        confidences = [self.run_syncnet(video) for video in videos]
        if len(confidences)!=0:
            count  = sum(map(lambda x : x>1, confidences))
            if (count/len(confidences))*100 >= 80:
                confidences = [2 if conf <= 1 else conf for conf in confidences]
            for i, conf in enumerate(confidences):
                if conf > 1:
                    speaker_data = {
                        "conf": conf,
                        "start_frame": speaker_frames[i][0],
                        "end_frame": speaker_frames[i][1],
                        "scene": n_scene,
                        "coords": speaker_coords[i],
                    }
                    lip_activity.append(speaker_data)
        return lip_activity

    def save_confs_to_json(self, data, scene_path):
        file_path = Path(scene_path, "lip_activity.json")
        if len(data) != 0:
            self.log.info(f"Saving info of the scene to '{file_path}'")
            with open(file_path, "w") as outfile:
                json.dump(data, outfile)

    def fix_overlaps_between_speakers(self, data):
        # More than 1 speaker detected, fix possible overlaps between speakers
        keys = list(data.keys())
        for speaker_1, speaker_2 in zip(keys, keys[1:]):
            if speaker_1 in data.keys() and speaker_2 in data.keys():
                speaker_2_deletes = 0
                speaker_1_deletes = 0
                frames_1 = [(info['start_frame'], info['end_frame']) for info in data[speaker_1]]
                confs_1 = [info['conf'] for info in data[speaker_1]]
                frames_2 = [(info['start_frame'], info['end_frame']) for info in data[speaker_2]]
                confs_2 = [info['conf'] for info in data[speaker_2]]
                for idx_1, frame_1 in enumerate(frames_1):
                    start_frame_1, end_frame_1 = frame_1
                    for idx_2, frame_2 in enumerate(frames_2):
                        start_frame_2, end_frame_2 = frame_2
                        if start_frame_1 >= start_frame_2:
                            is_between = start_frame_1 in range(start_frame_2, end_frame_2)
                        else:
                            is_between = start_frame_2 in range(start_frame_1, end_frame_1)
                        if is_between:
                            self.log.info(f"Found overlap between speakers: {', '.join(speaker for speaker in keys)}")
                            self.log.info("Speaker with the highest confidence will be kept.")
                            if confs_1[idx_1] > confs_2[idx_2]:
                                # speaker 2 with idx 2
                                if speaker_2 in data.keys():
                                    id_to_delete = max(idx_2 - speaker_2_deletes, 0)
                                    data[speaker_2].pop(id_to_delete)
                                    speaker_2_deletes += 1
                                if len(data[speaker_2])==0:
                                    data.pop(speaker_2)
                            else:
                                if speaker_1 in data.keys():
                                    id_to_delete = max(idx_1 - speaker_1_deletes, 0)
                                    data[speaker_1].pop(id_to_delete)
                                    speaker_1_deletes +=1
                                if len(data[speaker_1])==0:
                                    data.pop(speaker_1)
        return data
        
    def analyze_scene(self, faces, start_scene, end_scene):
        self.log.info("Finding the speaker(s) in the scene")
        speaker_confs = defaultdict(list)
        for speaker, coords_list in faces.items():
            speaker_frames, speaker_coords = self.divide_speeches_in_clips(coords_list)
            videos = []
            for j in range(len(speaker_frames)):                    
                video = self.video_pipeline.generate_video(speaker, speaker_frames[j], speaker_coords[j])
                videos.append(video)
            n_scene = f"{start_scene}-{end_scene}"
            lip_activity = self.detect_lip_activity(videos, n_scene, speaker_frames, speaker_coords)  
            if len(lip_activity) != 0:
                speaker_confs[speaker].extend(lip_activity)
        return speaker_confs

    def obtain_speeches(self, data):
        speeches = []
        if len(data.keys()) > 1:
            self.log.info("More than two speakers in the same scene. Looking into possible overlaps")
            data = self.fix_overlaps_between_speakers(data)
        for speaker, speaker_apps in data.items():
            coords = []
            for appearance in speaker_apps:
                coords.extend(appearance["coords"])
            info = {
                "start_frame": speaker_apps[0]["start_frame"],
                "end_frame": speaker_apps[-1]["end_frame"],
                "scene": speaker_apps[0]["scene"],
                "speaker": speaker,
                "coords": coords,
            }
            speeches.append(info)
        return speeches

    def transform_data(self, speaker, n_scene):
        speeches = []
        aux = list(speaker.values())[0]
        coords = [data[1] for data in aux]
        info = {
            "start_frame": aux[0][0],
            "end_frame": aux[-1][0],
            "scene": n_scene,
            "speaker": list(speaker.keys())[0],
            "coords": coords,
        }
        speeches.append(info)
        return speeches


    def run(self) -> None:
        frames, scenes = self.load_metadata(self.metadata_path)
        with alive_bar(len(scenes)-1) as bar:
            for start_scene, end_scene in zip(scenes, scenes[1:]):
                self.log.info(f"Detecting faces in scene: {start_scene}-{end_scene}")
                n_scene = f"{start_scene}-{end_scene}"
                scene_path = Path(self.tmp_path, n_scene)
                scene_path.mkdir(exist_ok=True)
                faces = self.detect_faces_in_scene(frames, start_scene, end_scene)
                if faces:
                    # Filter captured speakers that are less than 200 frames overall
                    faces = self.filter_short_videos(faces) 
                    faces = self.fix_missing_frames(faces)
                    self.log.info(f"Detected {len(faces)} appearance(s)")
                    self.video_pipeline = VideoCreatorPipeline(scene_path, self.video_path, self.session_name, start_scene, end_scene)
                    self.video_pipeline.obtain_frames(start_scene, end_scene)
                    if len(faces)!=1:
                        faces = self.analyze_scene(faces, start_scene, end_scene)
                        self.save_confs_to_json(faces, scene_path)
                        speeches = self.obtain_speeches(faces)
                    else:
                        speeches = self.transform_data(faces, n_scene)

                    aligned_scene_path = Path(self.aligned_path, n_scene)
                    a = AudioAlignmentPipeline(self.data_path, aligned_scene_path, self.session_name)
                    a.align_audio(speeches, start_scene, end_scene)
                    self.video_pipeline.clean_temporal_files()
                else:
                    self.log.info("No faces detected. Skipping scene")
                self.log.info("-------------")
                bar()

'''
def fix_overlaps_between_speakers(self, data):
        # More than 1 speaker detected, fix possible overlaps between speakers
        keys = list(data.keys())
        for speaker_1, speaker_2 in zip(keys, keys[1:]):
            if speaker_1 in data.keys() and speaker_2 in data.keys():
                speaker_2_deletes = 0
                speaker_1_deletes = 0
                frames_1 = [(info['start_frame'], info['end_frame']) for info in data[speaker_1]]
                confs_1 = [info['conf'] for info in data[speaker_1]]
                frames_2 = [(info['start_frame'], info['end_frame']) for info in data[speaker_2]]
                confs_2 = [info['conf'] for info in data[speaker_2]]
                for idx_1, frame_1 in enumerate(frames_1):
                    start_frame_1, end_frame_1 = frame_1
                    for idx_2, frame_2 in enumerate(frames_2):
                        start_frame_2, end_frame_2 = frame_2
                        if start_frame_1 >= start_frame_2:
                            is_between = start_frame_1 in range(start_frame_2, end_frame_2)
                        else:
                            is_between = start_frame_2 in range(start_frame_1, end_frame_1)
                        if is_between:
                            self.log.info(f"Found overlap between speakers: {', '.join(speaker for speaker in keys)}")
                            self.log.info("Speaker with the highest confidence will be kept.")
                            if confs_1[idx_1] > confs_2[idx_2]:
                                    # speaker 2 with idx 2
                                    if speaker_2 in data.keys():
                                        id_to_delete = max(idx_2 - speaker_2_deletes, 0)
                                        data[speaker_2].pop(id_to_delete)
                                        speaker_2_deletes += 1
                                    if len(data[speaker_2])==0:
                                        data.pop(speaker_2)
                            else:
                                    # speaker 2 with idx 2
                                    if speaker_1 in data.keys():
                                        id_to_delete = max(idx_1 - speaker_1_deletes, 0)
                                        data[speaker_1].pop(id_to_delete)
                                        speaker_1_deletes +=1
                                    if len(data[speaker_1])==0:
                                        data.pop(speaker_1)
'''