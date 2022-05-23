from fi_parliament_tools.pipeline import Pipeline
from alive_progress import alive_bar
from pathlib import Path
import json
from fi_parliament_tools.video_utils.create_videos import VideoCreatorPipeline
import math


class AudioAlignmentPipeline(Pipeline):
    def __init__(
        self, transcript_path: Path, data_video_path: str, output_path: Path
    ) -> None:
        self.transcript_path = transcript_path
        self.data_video_path = Path("data/processed/", data_video_path)
        self.output_path = output_path
        self.output_path.mkdir(exist_ok=True)
        self.session_name = data_video_path.split(".")[0]
        self.video_path = Path("data/raw/033-2020/session-033-2020.mp4")
        

    def read_json(self):
        with open(self.data_video_path, "r") as f:
            data = json.load(f)
        return data

    def read_transcript(self):
        transcript_data = []
        data = open(self.transcript_path, "r")
        for row in data:
            split_line = row.split(" ", 1)
            metadata, transcript = split_line
            split_metadata = metadata.split("-")
            video_id = "-".join(split_metadata[1:3])
            start_time, end_time = split_metadata[3:5]
            if video_id == self.session_name:
                sentence_data = {
                    "start_timestamp": start_time,
                    "end_timestamp": end_time,
                    "id": metadata,
                    "transcript": transcript,
                }
                transcript_data.append(sentence_data)
        return transcript_data

    def timestamp_to_frame(self, seconds):
        seconds = float(seconds) / 100
        n_frame = math.floor(float(seconds) * 25)
        return int(n_frame)

    def align_audio(self, data, transcript):
        for data_row in data:
            scene = data_row["scene"]
            start_scene, end_scene = scene.split('-')
            start_time_d = data_row["start_frame"]
            end_time_d = data_row["end_frame"]
            creator = VideoCreatorPipeline(self.output_path, self.video_path,  self.session_name, int(start_scene), int(end_scene))
            creator.obtain_frames(start_time_d, end_time_d)

            for trans_row in transcript:
                start_time_t = self.timestamp_to_frame(trans_row["start_timestamp"])
                end_time_t = self.timestamp_to_frame(trans_row["end_timestamp"])
                if start_time_t >= start_time_d and end_time_t <= end_time_d:
                    frames = (start_time_t, end_time_t)
                    start_index = start_time_t - start_time_d
                    end_index = end_time_t - start_time_d
                    coords = data_row["coords"][start_index:end_index]
                    path_scene = Path(self.output_path, scene)
                    path_scene.mkdir(exist_ok=True)
                    print(frames)
                    video = creator.generate_video(data_row["speaker"], frames, coords)
                    with open(video[:-4] + ".txt", "w") as f:
                        f.write(trans_row["transcript"])

    def run(self):
        data = self.read_json()
        transcript = self.read_transcript()
        self.align_audio(data, transcript)


transcript_path = Path("data/raw/text")
output_path = Path("data/aligned/")
data_video_path = "033-2020.json"
a = AudioAlignmentPipeline(transcript_path, data_video_path, output_path)
a.run()
