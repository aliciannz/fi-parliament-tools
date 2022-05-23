"""Command line client for face extraction Finnish Parliament data."""
from fi_parliament_tools.pipeline import Pipeline
from typing import List
from typing import Dict
from typing import Any
from pathlib import Path
import json
from collections import defaultdict
from typing import DefaultDict
import numpy as np
from alive_progress import alive_bar
from syncnet.run_syncnet import run_syncnet
from syncnet.run_pipeline import run_pipeline
from argparse import Namespace
import shutil
from fi_parliament_tools.video_utils.create_videos import VideoCreatorPipeline

 
class SpeakerExtractionPipeline(Pipeline):
    def __init__(
        self, metadata_path: str, video_path: str, output_path: str
    ) -> None:
        self.metadata_path = Path(metadata_path).resolve()
        self.video_path = Path(video_path).resolve()
        self.output_path = output_path
        self.output_path.mkdir(exist_ok=True)

        self.distance_threshold = 200  
        self.missing_threshold = 150
        self.cost_threshold = 30
        self.frames_per_clip = 200

        self.width_height = [1920, 1080]
        self.center_of_vid = [elem / 2 for elem in self.width_height]
        self.video_name = self.obtain_session_name(video_path)

    def load_metadata(self, path: Path):
        with path.open(mode="r", encoding="utf-8", newline="") as file:
            video_metadata = json.load(file)
            scenes = video_metadata["scenes"]
            frames = video_metadata["frames"]
            frames = {int(frame): values for frame, values in frames.items()}
        return frames, scenes

    def obtain_session_name(self, video_path):
        video_name = str(video_path).split("/")[-1]
        video_name = video_name[:-4]
        video_name = video_name.split("-", 1)[1]
        return video_name

    def obtain_center_bb(self, coords): 
        center_point = [coords[0] / 2 + coords[2], coords[1] / 2 + coords[3]]
        return center_point

    def find_centered_speakers(self, frame_data: List[Dict[str, Any]]):
        frame_coords = [speaker_info["coords"] for speaker_info in frame_data]
        conv_coords = [self.obtain_center_bb(coord) for coord in frame_coords]

        diff = np.array(conv_coords) - np.array(self.center_of_vid)
        distances = np.linalg.norm(diff, axis=1)  # type: ignore
        indices = np.argsort(distances)
        condition = (distances <= self.distance_threshold + distances[indices[:1]])
        centered_inds = np.where(condition)[0]
        speakers = [frame_data[index] for index in centered_inds]
        return speakers

    def compute_distances(self, speakers, faces):
        faces_coords = np.array([self.obtain_center_bb(data[-1][1]) for data in faces.values()])
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
            if len(coords_list) >= self.frames_per_clip:
                new_faces[speaker] = coords_list
        return new_faces

    def run_syncnet(self, video):
        model_path = Path("models", "syncnet_v2.model")
        ref_name = video.split("/")[-2:]
        ref_name = "/".join(ref_name)[:-4]
        tmp_path = Path(self.output_path, "syncnet_tmp")
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
            with open(file_path, "w") as outfile:
                json.dump(data, outfile)

    def read_jsons(self):
        json_paths = Path("data/processed/033-2020").glob("**/lip_activity.json")
        speeches = []
        for path in json_paths:
            with open(path, "r") as f:
                data = json.load(f)
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
        file_path = Path(self.output_path, f"{self.video_name}.json")
        with open(file_path, "w") as outfile:
            json.dump(speeches, outfile)

    def run(self) -> None:
        frames, scenes = self.load_metadata(self.metadata_path)
        path = Path(self.output_path, self.video_name)
        path.mkdir(exist_ok=True)
        with alive_bar(len(scenes)) as bar:
            for start_scene, end_scene in zip(scenes, scenes[1:]):
                print(f"Scene: {start_scene}-{end_scene}")
                scene_path = Path(path, f"{start_scene}-{end_scene}")
                scene_path.mkdir(exist_ok=True)
                faces = self.detect_faces_in_scene(frames, start_scene, end_scene)
                faces = self.filter_short_videos(faces) # Filter captured speakers that are less than 200 frames overall
                
                if faces:
                    speaker_confs = defaultdict(list)
                    self.video_pipeline = VideoCreatorPipeline(path, self.video_path, self.video_name, start_scene, end_scene)
                    self.video_pipeline.obtain_frames()

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

                    self.video_pipeline.clean_temporal_files()
                    self.save_confs_to_json(speaker_confs, scene_path)
                bar()


output_path = Path("data/processed")
metadata_path = Path("data/processed/033-2020/033-2020_faces.json")
video_path = Path("data/raw/033-2020/session-033-2020.mp4")

test = SpeakerExtractionPipeline(
    metadata_path=metadata_path, video_path=video_path, output_path=output_path
)
test.run()