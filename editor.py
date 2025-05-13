import cv2
import subprocess
import math
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from time import time
import re
import ffmpeg

from diarize import diarize
from faces import FaceIDModel
from gui import GUI
from subtitles import generate_word_srt, generate_sentence_srt, add_subtitles_from_srt 


# A handful of utility functions
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def make_json_serializable(data):
    serializable = {}
    for key, value in data.items():
        # Convert tuple keys to strings
        key_str = str(key)
        inner_dict = {}
        for inner_key, inner_val in value.items():
            # Convert tuple values to lists
            inner_dict[inner_key] = list(inner_val)
        serializable[key_str] = inner_dict
    return serializable


def stringify_keys(d):
    return {str(k): v for k, v in d.items()}


def remove_duplicates(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def delete_files_by_regex(directory, pattern):
    regex = re.compile(pattern)
    for file in Path(directory).iterdir():
        if file.is_file() and regex.match(file.name):
            file.unlink()


def parse_keys_to_tuples(d):
    def try_tuple(k):
        if k.startswith("(") and k.endswith(")"):
            try:
                return tuple(map(int, k[1:-1].split(", ")))
            except:
                pass
        return k
    return {try_tuple(k): v for k, v in d.items()}


class TikTokEditor:
    def __init__(self, video_path, n_speakers, embed_threshold, word_subtitles, delete_cache):
        # Ensure output and cache directories exist before using them
        os.makedirs("output", exist_ok=True)
        os.makedirs("cache", exist_ok=True)

        self.video_path = video_path
        self.video_title = Path(video_path).stem
        self.n_speakers = n_speakers
        self.embed_threshold = embed_threshold
        self.word_subtitles = word_subtitles

        # return frame rate fraction from ffmpeg probe
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        r_frame_rate = video_stream['r_frame_rate']  # e.g., "30000/1001"
        num, den = map(int, r_frame_rate.split('/'))

        # Delete files in cache corresponding to video_title if flag set
        if delete_cache:
            delete_files_by_regex("cache", self.video_title)

        self.output_path = os.path.join("output", "output.mp4")
        self.output_final_path = os.path.join("output", "output_final.mp4")
        self.output_final_subtitled_path = os.path.join("output", f"{self.video_title}_final_subtitled.mp4")
        # Create paths to cache files
        self.segments_path = os.path.join("cache", f"{self.video_title}_segments.json")
        self.blend_path = os.path.join("cache", f"{self.video_title}_blend.json")
        self.subtitle_path = os.path.join("cache", f"{self.video_title}.srt")
        self.face_db_path = os.path.join("cache", f"{self.video_title}_face_db.json")
        self.shots_path = os.path.join("cache", f"{self.video_title}_shots.json")
        self.ids_dict_path = os.path.join("cache", f"{self.video_title}_ids.json")
        self.ids_dict = {} # {speaker_id: face_id}

        # Create file that with video and cache info for easy importing in blender
        with open(os.path.join(str(Path.home()), ".last_video.txt"), "w") as f:
            f.write(str(Path(self.video_path).resolve()) + "\n")
            f.write(str(Path(self.blend_path).resolve()) + "\n")
            f.write(str(Path(self.subtitle_path).resolve()) + "\n")
            f.write(str(num) + "\n")
            f.write(str(den) + "\n")


    # Perform speaker diarization, face detection and embedding, 
    # and create gui for matching similar faces and matching faces to speakers
    def analyze(self, redo_faces):
        # Perform speaker diarization on video or load diarization cache
        if os.path.exists(self.segments_path):
            with open(self.segments_path, "r") as f:
                self.speaker_segments = json.load(f)
        else:
            self.speaker_segments = diarize(self.video_path, n_speakers=self.n_speakers)
            with open(self.segments_path, "w") as f:
                json.dump(self.speaker_segments, f, indent=4, default=convert_to_serializable)

        # Return face embeddings and segments separated by shot changes on video 
        # or load embeddings and segments from cache
        if os.path.exists(self.face_db_path) and os.path.exists(self.shots_path):
            with open(self.face_db_path, "r") as f:
                self.face_db = json.load(f)
            with open(self.shots_path, "r") as f:
                self.shot_segments = parse_keys_to_tuples(json.load(f))
        else:
            self.face_db, self.shot_segments = FaceIDModel(self.video_path, embed_threshold=self.embed_threshold).run()
            with open(self.face_db_path, "w") as f:
                json.dump(self.face_db, f, indent=4, default=convert_to_serializable)
            with open(self.shots_path, "w") as f:
                json.dump(stringify_keys(self.shot_segments), f, indent=4, default=make_json_serializable)

        # Open GUI for matching faces to speakers or load from cache
        if os.path.exists(self.ids_dict_path) and not redo_faces:
            with open(self.ids_dict_path, "r") as f:
                self.ids_dict = json.load(f)
        else:
            self.gui = GUI(self.video_path)
            self.gui.match_faces_to_voices(self.face_db, self.speaker_segments)
            self.combine_speakers_faces()
            with open(self.ids_dict_path, "w") as f:
                json.dump(self.ids_dict, f, indent=4)
            with open(self.shots_path, "w") as f:
                json.dump(stringify_keys(self.shot_segments), f, indent=4, default=make_json_serializable)


    # Iterates through shot_segments and combines faces and face data 
    # if they were linked in the match_faces_to_voices function from GUI 
    def combine_speakers_faces(self):
        combine_ids_dict = {}

        for speaker_id, face_ids in self.gui.speakers_to_faces.items():
            if len(face_ids) == 1:
                combine_ids_dict[face_ids[0]] = None
            else:
                combine_ids_dict[face_ids[0]] = set(face_ids[1:])

        main_keys = set(combine_ids_dict.keys())

        for speaker_id, face_ids in self.gui.speakers_to_faces.items():
            for id in face_ids:
                if id in main_keys:
                    self.ids_dict[speaker_id] = id

        for segment, face_db in self.shot_segments.items():
            new_db = {}
            for main_id, other_ids in combine_ids_dict.items():
                if main_id in face_db:
                    if main_id not in combine_ids_dict:
                        continue
                    new_db[main_id] = face_db[main_id]
                    if other_ids is not None:
                        for id in other_ids:
                            if id in face_db:
                                new_db[main_id] = tuple(x + y for x, y in zip(new_db[main_id], face_db[id])) 
            for id, v in new_db.items():
                new_db[id] = (v[1]/v[0], v[2]/v[0])
            self.shot_segments[segment] = new_db


    def create_subtitles(self):
        if self.word_subtitles:
            generate_word_srt(self.video_path, self.subtitle_path)
        else:
            generate_sentence_srt(self.video_path, self.subtitle_path)


    def add_subs_to_video(self):
        add_subtitles_from_srt(self.output_final_path, self.subtitle_path, self.output_final_subtitled_path)
        os.remove(self.output_final_path)


    # Create blend cache file for importing to blender
    # contains position of where to set frame for every frame in the video
    def prepare_for_blender(self, add_subtitles, new_subs):
        start_time = time()
        cap = cv2.VideoCapture(self.video_path)

        # Get video properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Calculate new width for 9:16 aspect ratio
        new_width = int(math.ceil(height * (9 / 16)))
        output_size = (new_width, height)

        with open(os.path.join(str(Path.home()), ".last_video.txt"), "a") as f:
            f.write(str(new_width) + "\n")
            f.write(str(height) + "\n")

        blend = []

        timeline_index = 0
        for segment, face_db in self.shot_segments.items():
            start, end = segment
            boxes = {}
            for id, box in face_db.items():
                box = (box[0] - new_width//2, box[0] + new_width//2)
                boxes[id] = box

            for i in range(start, end + 1):
                time_sec = (i - 1) / fps

                # Determine current speaker
                while timeline_index < len(self.speaker_segments):
                    speaker_id, s, e = self.speaker_segments[timeline_index]
                    if s <= time_sec <= e:
                        current_speaker = speaker_id
                        break
                    elif time_sec > e:
                        timeline_index += 1
                    else:
                        break

                # edit by camera switching
                # edit by diarization if more than one face in shot
                # if only one face on screen, default to singular face
                # (helps with choppy diarization from quick speaking etc.)
                ids_list = list(face_db.keys())
                if ids_list:
                    if len(ids_list) == 1:
                        face_id = list(face_db.keys())[0] 
                    else:
                        face_id = self.ids_dict.get(current_speaker)
                    bbox = boxes.get(face_id)
                else:
                    bbox = False

                if bbox:
                    x1, x2 = bbox
                else:
                    # default to middle of screen if no faces
                    x1, x2 = width//2 - new_width//2, width//2 + new_width//2

                # Handle cases where faces are too close to edge
                if x2 > width:
                    x1 = width-new_width
                    x2 = int(width)
                elif x1 < 0:
                    x1 = 0
                    x2 = new_width

                blend.append((i, 0, height, int(x1), int(x2)))

        blend = remove_duplicates(blend)
        with open(self.blend_path, "w") as f:
            json.dump(blend, f, indent=4)

        print(f"blend.json for video was created in {time() - start_time:.2f}s")

        if new_subs and os.path.exists(self.subtitle_path):
            os.remove(self.subtitle_path)

        if add_subtitles:
            self.create_subtitles()







    def edit(self):
        self.crop_video_on_speaker_bbox_static()
        self.extract_audio_and_apply_to_video()
        subprocess.run(["mpv", self.output_final_path, "--volume=60"])


    def edit_w_subtitles(self, blender_prep):
        self.crop_video_on_speaker_bbox_static()
        self.extract_audio_and_apply_to_video()
        self.add_subs_to_video()
        subprocess.run(["mpv", self.output_final_subtitled_path, "--volume=60"])


    def crop_video_on_speaker_bbox_static(self):
        cap = cv2.VideoCapture(self.video_path)

        # Get video properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate new width for 9:16 aspect ratio
        new_width = int(math.ceil(height * (9 / 16)))
        output_size = (new_width, height)

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, output_size)

        timeline_index = 0
        for segment, face_db in self.shot_segments.items():
            start, end = segment
            boxes = {}
            for id, box in face_db.items():
                box = (box[0] - new_width//2, box[0] + new_width//2)
                boxes[id] = box

            for i in tqdm(range(start, end + 1)):
                ret, frame = cap.read()
                if not ret:
                    break

                time_sec = (i - 1) / fps

                # Determine current speaker
                while timeline_index < len(self.speaker_segments):
                    speaker_id, start, end = self.speaker_segments[timeline_index]
                    if start <= time_sec <= end:
                        current_speaker = speaker_id
                        break
                    elif time_sec > end:
                        timeline_index += 1
                    else:
                        break

                # edit by camera switching
                # edit by diarization if more than one face in shot
                # if only one face on screen, default to singular face
                # (helps with choppy diarization from quick speaking etc.)
                ids_list = list(face_db.keys())
                if ids_list:
                    if len(ids_list) == 1:
                        face_id = list(face_db.keys())[0] 
                    else:
                        face_id = self.ids_dict.get(current_speaker)
                    bbox = boxes.get(face_id)
                else:
                    bbox = False

                if bbox:
                    x1, x2 = bbox
                else:
                    # default to middle of screen if no faces
                    x1, x2 = width//2 - new_width//2, width//2 + new_width//2

                # Handle cases where faces are too close to edge
                if x2 > width:
                    cropped = frame[0:height, -new_width:]
                elif x1 < 0:
                    cropped = frame[0:height, 0:new_width]
                else:
                    cropped = frame[0:height, int(x1):int(x2)]

                out.write(cropped)
        cap.release()
        out.release()
        print("Done writing:", self.output_path)


    def extract_audio_and_apply_to_video(self):
        """
        Extracts audio from the input video file and applies it to a new video file using FFmpeg.

        """
        # Step 1: Extract audio from the input video
        audio_file = "extracted_audio.aac"  # Temporary file for extracted audio
        extract_audio_command = [
            'ffmpeg', '-loglevel', 'quiet', 
            '-i', self.video_path,  # Input video file
            '-vn',  # No video stream
            '-acodec', 'aac',  # Audio codec (AAC)
            '-strict', 'experimental',  # Allow experimental codecs
            audio_file  # Output audio file
        ]
        
        # Run the extract audio command
        subprocess.run(extract_audio_command)
        print(f"Audio extracted to {audio_file}")
        
        # Step 2: Apply the extracted audio to the new video
        apply_audio_command = [
            'ffmpeg', '-loglevel', 'quiet', 
            '-i', self.output_path,  # Input new video file
            '-i', audio_file,  # Input extracted audio file
            '-c:v', 'copy',  # Copy video stream (no re-encoding)
            '-c:a', 'aac',  # Encode audio as AAC
            '-strict', 'experimental',  # Allow experimental codecs
            '-map', '0:v:0',  # Map the first video stream
            '-map', '1:a:0',  # Map the first audio stream
            self.output_final_path  # Output file path
        ]
        
        # Run the apply audio command
        subprocess.run(apply_audio_command)
        print(f"Final video with audio saved to {self.output_final_path}")
        
        # Optional: Remove the temporary audio file after processing
        os.remove(audio_file)
        print(f"Temporary audio file {audio_file} removed.")
        os.remove(self.output_path)
