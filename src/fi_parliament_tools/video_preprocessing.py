"""Functionality for extracting face crops from parliament videos."""
from collections import defaultdict
from typing import Dict, List
from fi_parliament_tools.pipeline import Pipeline
from typing import Any
from typing import DefaultDict
from pathlib import Path
import json
import subprocess
from alive_progress import alive_bar
from logging import Logger

class VideoPreprocessingPipeline(Pipeline):
    def __init__(self, log: Logger, metadata_path: str, output_path: str) -> None:
        super().__init__(log)
        self.metadata_path = Path(metadata_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.output_path.mkdir(exist_ok=True)

    def obtain_scene_changes(self, scene_changes_path: Path, first_frame: int):
        scene_changes_files = scene_changes_path.glob("*.json")
        scene_changes = [first_frame]
        files = [file for file in scene_changes_files if file.is_file()]
        for file in files:
            with file.open(mode="r", encoding="utf-8", newline="") as scenes:
                data = json.load(scenes)
                scene_changes.extend(data["frame_indices"])
        return sorted(scene_changes)

    def obtain_features(self, features_path: Path):
        features_files = features_path.glob("*.jsonl")
        files = [file for file in features_files if file.is_file()]
        features = defaultdict(list)
        for file in files:
            with file.open(mode="r", encoding="utf-8", newline="") as feats:
                data = [json.loads(line) for line in feats]
                for element in data:
                    features[element['frame']].append({'coords':element['box'], 'keypoints': element['keypoints']})

        return features


    def modify_coords(self, face: List[int]) -> List[int]:
        """Compute coordinates for the face crops.

        The coordinates in the input list are in the order:
        top left x, top left y, right bottom x, right bottom y

        Args:
            face (List[str]): bounding box coordinates for the face detection

        Raises:
            ValueError: if face crop is size zero or less

        Returns:
            List[int]: image coordinates for the face crop
        """
        width = face[2] - face[0]
        height = face[3] - face[1]

        if width <= 0 or height <= 0:
            raise ValueError("Face crop width or height is less than 0.")
        face_coords = [width, height, face[0], face[1]]
        return face_coords

    def find_face_features(self, speaker_coords, features_coords):
        for features in features_coords:
            if features['coords'] == speaker_coords:
                return features['keypoints']
        return []

    def read_video_metadata(
        self, metadata_path: Path, features
    ) -> DefaultDict[int, List[Dict[str, Any]]]:
        """Reads the metadata provided to obtain the frame numbers, the speaker id's per each frame and the coordinates of each speaker per frame

        Args:
            metadata (TextIO): metadata of the video

        Returns:
            (DefaultDict[int, List[Dict[str, Any]]]): dictionary containing the speakers with their frames and coordinates
        """
        frames = defaultdict(list)
        with metadata_path.open(mode="r", encoding="utf-8", newline="") as metadata:
            for row in metadata:
                row_data = row.split(" ")
                frame_number = int(row_data[2])
                speaker_id = row_data[-1].rstrip()
                speaker_coords = [int(coord) for coord in row_data[6:10]]
                mod_speaker_coords = self.modify_coords(speaker_coords)
                speaker_features = self.find_face_features(speaker_coords, features[frame_number])
                speaker_dict = {"coords": mod_speaker_coords, "speaker_id": speaker_id, "speaker_features": speaker_features}
                frames[frame_number].append(speaker_dict)
        return frames

    def save_to_json(self, session, frames, scenes):
        json_path = Path(self.output_path, f"{session}")
        json_path.mkdir(exist_ok=True)
        video = {
            "width_height": [1920, 1080],
            "fps": 25,
            "frames": frames,
            "scenes": scenes,
        }
        with open(Path(json_path, f"{session}_faces.json"), "w") as fp:
            json.dump(video, fp)
        return json_path

    def read_directories(self):
        sessions_paths = []
        for path in self.metadata_path.iterdir():
            if path.is_dir():
                session_name = str(path).split("/")[-1]
                data = Path(path, f"session-{session_name}-faces.txt")
                scenes_path = Path(path, "scene_changes/")
                features_path = Path(path, "features/")
                video_path = Path(path, f"session-{session_name}.mp4")
                sessions_paths.append(
                    {
                        "session_name": session_name,
                        "video_path": video_path,
                        "metadata": data,
                        "scenes_path": scenes_path,
                        "features_path": features_path
                    }
                )
        return sessions_paths

    def run(self) -> None:
        sessions_paths = self.read_directories()
        self.log.info(f"Found {len(sessions_paths)} parliament videos")
        with alive_bar(len(sessions_paths)) as bar:
            for session in sessions_paths:
                self.log.info(f"Preprocessing session '{session['session_name']}'.")
                features = self.obtain_features(session["features_path"])
                frames = self.read_video_metadata(session["metadata"], features)
                first_frame = min(frames.keys())
                scenes = self.obtain_scene_changes(session["scenes_path"], first_frame)
                save_path = self.save_to_json(session["session_name"], frames, scenes)
                self.log.info(f"Saving session data in '{save_path}'")
            bar()