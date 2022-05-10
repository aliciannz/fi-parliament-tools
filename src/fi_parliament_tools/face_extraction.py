"""Functionality for extracting face crops from parliament videos."""
from collections import defaultdict
from typing import Dict, List
from typing import TextIO
from typing import Tuple
from typing import DefaultDict
from typing import Any
import numpy as np
import os


# find corpus/ -iname "*.mp4" | sort > video_files.list

fps = 25
vid_dim = (1920, 1080)
center_of_vid = (vid_dim[0] / 2, vid_dim[1] / 2)
scale = "60:80"


def compute_coords(face: List[str]) -> List[int]:
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
    width = int(face[2]) - int(face[0])
    height = int(face[3]) - int(face[1])
    if width <= 0 or height <= 0:
        raise ValueError("Face crop width or height is less than 0.")
    face_coords = [int(width), int(height), int(face[0]), int(face[1])]
    return face_coords

def find_nearest_face(coords: List[List[int]]) -> int:
    """Given a list of faces' coordinates, computes the nearest face to the center of the video using the euclidean distance

    Args:
        coords (List[List]): List of list containing the coordinates of the faces of a frame

    Returns:
        int: the index containing the nearest coordinates to the center of the video
    """
    converted_coords = [(coord[0] / 2 + coord[2], coord[1] / 2 + coord[3]) for coord in coords]
    distances = np.linalg.norm(np.array(converted_coords) - np.array(center_of_vid), axis=1) # type: ignore
    min_index = np.argmin(distances)
    return int(min_index)


def convert_to_timestamp(frame: int) -> str:
    """Convert a frame number to timestamp of seconds and milliseconds in string format.

    Args:
        frame (int): frame number

    Returns:
        str: string containing the seconds and milliseconds of a frame number
    """
    milliseconds = 1000 * frame / fps
    seconds, milliseconds = divmod(milliseconds, 1000)
    return "{:02d}.{:03d}".format(int(seconds), int(milliseconds))



def extract_centered_speaker(frame_data: List[Dict[str, Any]]) -> Tuple[str, List[int]] :
    """Extract the most centered speaker with respect to the center of the video

    Args:
        frame_data (List[Dict[str, Any]]): frame data that contains the speakers and their coordinates

    Returns:
        (Tuple[str, List[int]]): speaker's id and its coordinates
    """
    coords_of_frame = [speaker_info['coords'] for speaker_info in frame_data]
    # coords_of_frame = [coords for speaker_id, coords in frame_data] mypy cant infer this
    centered_face_index = find_nearest_face(coords_of_frame)

    # Extract speaker name and coords of the nearest face to video's center
    curr_speaker = frame_data[centered_face_index]['speaker_id']
    curr_speaker_coords = frame_data[centered_face_index]['coords']
    return curr_speaker, curr_speaker_coords


def create_script(current_speaker: str, speeches_by_speaker: List[str], range_frames: str, script: List[str], ffmpeg_ids: List[str]) -> None:
    """Create script [...]

    Args:
        current_speaker (str): the current speaker's id
        speeches_by_speaker (List[str]): _description_
        range_frames (str): range of frames used in the script
        script (List[str]): list containing all the commands for the ffmpeg script
        ffmpeg_ids (List[str]): all the ffmpeg ids used to create a script
    """
    total_splits = len(ffmpeg_ids) / 2
    n_clips, remainder = divmod(total_splits, 100) 
    n_clips = int(n_clips)
    remainder = int(remainder)
    range_of_values = range(0, n_clips)

    for i in range(0, n_clips):
        if i == range_of_values[-1]:
            elements = ffmpeg_ids[i * 200 : i * 200 + 200 + remainder * 2]
            curr_script = script[i * 300 : i * 300 + 300 + remainder * 3]

        else:
            elements = ffmpeg_ids[i * 200 : i * 200 + 200]
            curr_script = script[i * 300 : i * 300 + 300]

        last_part = "".join(
            "[{}]".format(i) for i in elements
        ) + "concat=n={}:v=1:a=1[video][audio]".format(len(elements) / 2)
        script_text = "".join(curr_script)
        total_script = script_text + last_part
        speech = speeches_by_speaker.count(current_speaker)
        path = os.path.join("speeches", range_frames)

        os.makedirs(path, exist_ok=True)
        name_file = f"{path}/{current_speaker}_{speech}_{i}"
        text_file = open(name_file + ".txt", "w")
        text_file.write(total_script)
        text_file.close()
        command = 'ffmpeg -i session-033-2020.mp4 -filter_complex_script {}.txt -map "[video]":v -map "[audio]" -threads 6 -c:v libx264 -preset veryslow {}.mp4'.format(
            name_file, name_file
        )
        os.system(command)


def generate_clips_of_speakers(frames:  DefaultDict[int, List[Dict[str, Any]]]) -> None:
    """Extract the most centered speakers of each frame and generate mp4 clips of 4 seconds [...]

    Args:
        frames (DefaultDict[int, List[Dict[str, Any]]]): dictionary containing the frames and the speakers and their coordinates
    """
    speeches_by_speaker: List[str] = []
    script: List[str] = []
    ffmpeg_ids: List[str] = []
    frame_numbers = list(sorted(frames.keys()))
    starting_frame = frame_numbers[0]

    for id_frame, frame in enumerate(frame_numbers):
        frame_data = frames[frame]
        current_speaker, speaker_coords = extract_centered_speaker(frame_data)
        # Convert frame numbers to timestamps
        first_timestamp = convert_to_timestamp(frame)
        last_timestamp = convert_to_timestamp(frame + 1)

        # Generate ffmpeg commands for the current frame
        script.append(
            "[0:v]trim=start={}:end={},setpts=PTS-STARTPTS [v{}]; \n".format(
                first_timestamp, last_timestamp, id_frame
            )
        )
        script.append(
            "[0:a]atrim=start={}:end={},asetpts=PTS-STARTPTS [a{}];\n".format(
                first_timestamp, last_timestamp, id_frame
            )
        )
        coords_str = ":".join([str(val) for val in speaker_coords])
        script.append(
            "[v{}]crop={}, scale={} [b{}]; \n".format(id_frame, coords_str, scale, id_frame)
        )

        ffmpeg_ids.append("b" + str(id_frame))
        ffmpeg_ids.append("a" + str(id_frame))

        if id_frame + 1 == len(frame_numbers):
            next_speaker = "None"
            next_frame = -1
        else:
            next_frame_data = frames[frame_numbers[id_frame + 1]]
            next_speaker, _ = extract_centered_speaker(next_frame_data)
            next_frame = frame_numbers[id_frame + 1]

        if current_speaker != next_speaker or frame + 1 != next_frame:
            range_frames = "{}-{}".format(starting_frame, frame)
            create_script(current_speaker, speeches_by_speaker, range_frames, script, ffmpeg_ids)
            ffmpeg_ids = []
            script = []
            speeches_by_speaker.append(current_speaker) 
            starting_frame = next_frame


def read_video_metadata(metadata: TextIO) -> DefaultDict[int, List[Dict[str, Any]]]:
    """Reads the metadata provided to obtain the frame numbers, the speaker id's per each frame and the coordinates of each speaker per frame

    Args:
        metadata (TextIO): metadata of the video

    Returns:
        (DefaultDict[int, List[Dict[str, Any]]]): dictionary containing the speakers with their frames and coordinates
    """
    frames = defaultdict(list)
    for row in metadata:
        info_row = row.split(" ")
        speaker_id = info_row[-1].rstrip()
        frame_number = int(info_row[2])
        coords = compute_coords(info_row[6:10])
        dict_values = {'coords': coords, 'speaker_id': speaker_id}
        frames[frame_number].append(dict_values)
    return frames

if __name__ == "__main__":
    metadata = open("session-033-2020-faces.txt", "r")
    frames = read_video_metadata(metadata)
    generate_clips_of_speakers(frames)
