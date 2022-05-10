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
import multiprocessing
import os
import shutil
import glob
from fi_parliament_tools.video_utils.create_videos import VideoCreatorPipeline
import time


class SpeakerExtractionPipeline(Pipeline):
    """
    TODO: Comments
    """
    def __init__(
        self, metadata_path: Path, video_path: Path, output_path: Path
    ) -> None:
        # super().__init__(log)
        # self.log = log
        self.metadata_path = Path(metadata_path).resolve()
        self.video_path = Path(video_path).resolve()
        self.output_path = output_path
        self.output_path.mkdir(exist_ok=True)

        self.threshold = 200  # TODO: Establish a threshold
        self.frames_per_clip = 200

        self.width_height = [1920, 1080]
        self.center_of_vid = [elem / 2 for elem in self.width_height]

        self.video_name = self.obtain_session_name(video_path)

        frames_path = os.path.join(f"data/processed/{self.video_name}/frames", "*.jpg")
        self.flist = glob.glob(frames_path)
        self.flist.sort()

        self.video_pipeline = VideoCreatorPipeline(frames_path, video_path)

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

    def find_centered_speakers(self, frame_data: List[Dict[str, Any]]):
        frame_coords = [speaker_info["coords"] for speaker_info in frame_data]
        converted_coords = [
            (coord[0] / 2 + coord[2], coord[1] / 2 + coord[3]) for coord in frame_coords
        ]
        diff = np.array(converted_coords) - np.array(self.center_of_vid)
        distances = np.linalg.norm(diff, axis=1)  # type: ignore
        indices = np.argsort(distances)
        center_faces_inds = np.where(
            distances <= self.threshold + distances[indices[:1]]
        )[0]
        speakers = [frame_data[index] for index in center_faces_inds]
        return speakers

    def detect_faces_in_scene(
        self,
        frames: DefaultDict[int, List[Dict[str, Any]]],
        start_scene: int,
        end_scene: int,
    ) -> None:
        faces = defaultdict(list)
        last_frame = start_scene
        speaker_cuts = []
        last_speaker_used = {}
        for frame in range(start_scene, end_scene):
            if frame in frames.keys() and frame == last_frame + 1:
                frame_data = frames[frame]
                speakers = self.find_centered_speakers(frame_data)
                for speaker in speakers:
                    name = speaker["speaker_id"]
                    frame_coords = (frame, speaker["coords"])
                    if name in faces.keys():
                        if speaker_cuts.count(name) == 0:
                            name_to_use = name
                            last_frame_speaker = faces[name][-1]
                        else:
                            name_to_use = last_speaker_used[name]
                            last_frame_speaker = faces[name_to_use][-1]

                        if last_frame_speaker[0] == frame - 1:
                            faces[name_to_use].append(frame_coords)
                            last_speaker_used[name] = name_to_use
                        else:
                            speaker_cuts.append(name)
                            new_name = f"{name}_{speaker_cuts.count(name)}"
                            faces[new_name].append(frame_coords)
                            last_speaker_used[name] = new_name
                    else:
                        faces[name].append(frame_coords)
            last_frame = frame
        return faces

    def filter_short_videos(self, faces):
        new_faces = defaultdict(list)
        for face in faces.items():
            speaker, coords_list = face
            if len(coords_list) > self.frames_per_clip:
                new_faces[speaker] = coords_list
        return new_faces

    def detect_lip_activity(self, video):
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

    def save_lip_activity(self, videos, n_scene, speaker_frames, speaker_coords):
        aux = []
        for i, video in enumerate(videos):
            conf = self.detect_lip_activity(video)
            if conf > 1:
                xx = {
                    "conf": conf,
                    "start_frame": speaker_frames[i][0],
                    "end_frame": speaker_frames[i][1],
                    "scene": n_scene,
                    "coords": speaker_coords[i],
                }
                aux.append(xx)
        return aux

    def divide_speeches(self, coords):
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

    def save_conf_to_json(self, data, scene_path):
        file_path = Path(scene_path, "lip_activity.json")
        if len(data) != 0:
            with open(file_path, "w") as outfile:
                json.dump(data, outfile)

    def run(self) -> None:
        frames, scenes = self.load_metadata(self.metadata_path)
        path = Path(self.output_path, self.video_name)
        path.mkdir(exist_ok=True)

        with alive_bar(int(len(scenes) / 2)) as bar:
            for start_scene, end_scene in zip(scenes, scenes[1:]):
                scene_path = Path(path, f"{start_scene}-{end_scene}")
                scene_path.mkdir(exist_ok=True)
                faces = self.detect_faces_in_scene(frames, start_scene, end_scene)
                faces = self.filter_short_videos(faces)  # !! Fix this.
                speaker_confs = defaultdict(list)

                for speaker, coords_list in faces.items():
                    speaker_frames, speaker_coords = self.divide_speeches(coords_list)
                    videos = []
                    for j in range(len(speaker_frames)):
                        video = self.video_pipeline.generate_audio_video(
                            scene_path, speaker, speaker_frames[j], speaker_coords[j]
                        )
                        videos.append(video)
                    print("Detecting lip activity")
                    n_scene = f"{start_scene}-{end_scene}"
                    lip_activity = self.save_lip_activity(
                        videos, n_scene, speaker_frames, speaker_coords
                    )

                    if len(lip_activity) != 0:
                        speaker_confs[speaker].extend(lip_activity)
                self.save_conf_to_json(speaker_confs, scene_path)#
            bar()

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


output_path = Path("data/processed")
metadata_path = Path("data/processed/033-2020/033-2020_faces.json")
video_path = Path("data/raw/033-2020/session-033-2020.mp4")
test = SpeakerExtractionPipeline(
    metadata_path=metadata_path, video_path=video_path, output_path=output_path
)
test.run()
#test.read_jsons()