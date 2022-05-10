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


# find corpus/ -iname "*.mp4" | sort > video_files.list

class VideoPreprocessingPipeline(Pipeline):
    def __init__(self, metadata_path: Path, output_path: Path , extract_frames: bool = True) -> None:
        self.metadata_path = metadata_path
        self.output_path = output_path
        self.output_path.mkdir(exist_ok=True)
        self.extract_frames = extract_frames

    def obtain_scene_changes(self, scene_changes_path: Path, frames):
        scene_changes_files = scene_changes_path.glob("*.json")
        # First frame can only be found in frames object
        first_frame = min(frames.keys())
        scene_changes = [first_frame]
        files = [file for file in scene_changes_files if file.is_file()]
        for file in files:
            with file.open(mode="r", encoding="utf-8", newline="") as scenes:
                data = json.load(scenes)
                scene_changes.extend(data["frame_indices"])
        return sorted(scene_changes)

    def obtain_frames(self, session, video_path):
        frames_path = Path(self.output_path, session, "frames")
        frames_path.mkdir(exist_ok=True)
        command = f"ffmpeg -y -i {video_path} -qscale:v 2 -threads 1 -f image2 {Path(frames_path, '%06d.jpg')}"
        subprocess.call(command, shell=True, stdout=None)

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

    def read_video_metadata(
        self, metadata_path: Path
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
                new_speaker_coords = self.modify_coords(speaker_coords)
                speaker_dict = {"coords": new_speaker_coords, "speaker_id": speaker_id}
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

    def read_directories(self):
        sessions_paths = []
        for path in self.metadata_path.iterdir():
            if path.is_dir():
                session_name = str(path).split("/")[-1]
                data = list(path.glob("session-*-faces.txt"))[0]  # TODO: Decide about this
                scene_changes = list(path.glob("scene_changes/"))[0]
                video_path = list(path.glob("session-*.mp4"))[0]
                sessions_paths.append(
                    {
                        "session_name": session_name,
                        "video_path": video_path,
                        "metadata": data,
                        "scene_changes": scene_changes,
                    }
                )
        return sessions_paths

    def run(self) -> None:
        sessions_paths = self.read_directories()
        with alive_bar(len(sessions_paths)) as bar:
            for session in sessions_paths:
                frames = self.read_video_metadata(session["metadata"])
                scenes = self.obtain_scene_changes(session["scene_changes"], frames)
                self.save_to_json(session["session_name"], frames, scenes)
                if self.extract_frames:
                    self.obtain_frames(session["session_name"], session["video_path"])
            bar()

if __name__ == "__main__":
    metadata_path = Path("data/raw")
    output_path = Path("data/processed")
    s = VideoPreprocessingPipeline(metadata_path, output_path)
    s.run()
