import cv2
import subprocess
from fi_parliament_tools.pipeline import Pipeline
from pathlib import Path
import glob
import os

class VideoCreatorPipeline(Pipeline):
    def __init__(self, frames_path: Path, video_path: Path) -> None:
        # self.video_name = "033-2020"
        self.flist = glob.glob(frames_path)
        self.flist.sort()
        self.video_path = video_path
        self.size = 224 #!!
        self.fps = 25  #!!

    def add_padding(self, face):
        width = max(self.size, face[0])
        height = max(self.size, face[1])
        x = face[2] - 60 #!!
        y = face[3] - 60 #!!
        if width <= 0 or height <= 0:
            raise ValueError("Face crop width or height is less than 0.")
        face_coords = [width, height, x, y]
        return face_coords

    def convert_to_timestamp(self, frame) -> str:
        milliseconds = 1000 * frame / self.fps
        seconds, milliseconds = divmod(milliseconds, 1000)
        return "{:02d}.{:03d}".format(int(seconds), int(milliseconds))

    def generate_audio(self, scene_path, start_frame, end_frame):
        audio_start = self.convert_to_timestamp(start_frame)
        audio_end = self.convert_to_timestamp(end_frame)
        audio_file = Path(scene_path, "audio.wav")
        command = f"ffmpeg -y -i {self.video_path} -ss {audio_start} -to {audio_end} {audio_file}"
        try:
            subprocess.run(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            msg = f"ffmpeg returned non-zero exit status {e.returncode}. Stderr:\n {e.stderr}"
            print(msg)
        return audio_file

    def combine_audio_video(self, audio_file, video_file):
        output_video = str(video_file) + ".avi"
        command = f"ffmpeg -y  -i {str(video_file)+'_tmp.avi'} -i {audio_file} -c:v copy -c:a copy {output_video}"
        try:
            subprocess.run(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            msg = f"ffmpeg returned non-zero exit status {e.returncode}. Stderr:\n {e.stderr}"
            print(msg)
        return output_video

    def generate_audio_video(self, total_path, speaker, frames_speaker, coords_speaker):
        codec = cv2.VideoWriter_fourcc(*"XVID")
        start_frame, end_frame = frames_speaker

        name_video = Path(total_path, f"{speaker}_{start_frame}_{end_frame}")
        tmp_video = Path(str(name_video) + "_tmp.avi")

        video_writer = cv2.VideoWriter(
            str(tmp_video), codec, self.fps, (self.size, self.size)
        )
        for index, n_frame in enumerate(range(start_frame, end_frame)):
            coords = self.add_padding(coords_speaker[index])
            w, h, x, y = coords
            image = cv2.imread(self.flist[n_frame])
            frame = image[y : y + h, x : x + w]
            video_writer.write(frame)
        video_writer.release()
        audio_file = self.generate_audio(total_path, start_frame, end_frame)
        
        video_file = self.combine_audio_video(audio_file, name_video)
        # We only need the video file
        audio_file.unlink()
        tmp_video.unlink()
        return video_file
