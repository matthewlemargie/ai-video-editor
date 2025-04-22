import cv2
import subprocess
import math
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from gui import GUI
from diarize import diarize
from faces import create_face_ids
from subtitles import generate_word_srt, generate_sentence_srt, add_subtitles_from_srt 


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class TikTokEditor:
    def __init__(self, video_path, n_speakers, max_num_faces, show_video, word_subtitles):
        os.makedirs("output", exist_ok=True)
        os.makedirs("segments_cache", exist_ok=True)
        os.makedirs("subtitles_cache", exist_ok=True)
        # os.makedirs("face_db_cache", exist_ok=True)

        self.n_speakers = n_speakers
        self.max_num_faces = max_num_faces
        self.show_video = show_video
        self.video_path = video_path
        self.word_subtitles = word_subtitles
        self.video_title = Path(video_path).stem
        self.output_path = os.path.join("output", "output.mp4")
        self.output_final_path = os.path.join("output", "output_final.mp4")
        self.output_final_subtitled_path = os.path.join("output", f"{self.video_title}_final_subtitled.mp4")
        self.segments_path = os.path.join("segments_cache", f"{Path(self.video_path).stem}_segments.json")
        self.subtitle_path = os.path.join("subtitles_cache", f"{self.video_title}.srt")
        # self.face_db_path = os.path.join("face_db_cache", f"{self.video_title}_face_db.json")
        self.ids_dict = {}


    def analyze(self):
        if os.path.exists(self.segments_path):
            with open(self.segments_path, "r") as f:
                self.speaker_segments = json.load(f)
        else:
            self.speaker_segments = diarize(self.video_path, n_speakers=self.n_speakers)
            with open(self.segments_path, "w") as f:
                json.dump(self.speaker_segments, f, indent=4, default=convert_to_serializable)

        self.face_db, self.shot_segments = create_face_ids(self.video_path, max_num_faces=self.max_num_faces, show_video=self.show_video)

        self.gui = GUI(self.video_path)
        self.gui.match_faces_to_voices(self.face_db.keys(), self.face_db, self.speaker_segments)
        self.combine_speakers_faces()


    def edit(self):
        self.crop_video_on_speaker_bbox_static()
        self.extract_audio_and_apply_to_video()
        subprocess.run(["mpv", self.output_final_path, "--volume=60"])


    def edit_w_subtitles(self):
        self.crop_video_on_speaker_bbox_static()
        self.extract_audio_and_apply_to_video()
        self.create_subtitle_video()
        subprocess.run(["mpv", self.output_final_subtitled_path, "--volume=60"])


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

        self.shot_segments = list(self.shot_segments.items())

        timeline_index = 0
        for segment, face_db in self.shot_segments:
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


    def create_subtitle_video(self):
        if self.word_subtitles:
            generate_word_srt(self.output_final_path, self.subtitle_path)
        else:
            generate_sentence_srt(self.output_final_path, self.subtitle_path)

        add_subtitles_from_srt(self.output_final_path, self.subtitle_path, self.output_final_subtitled_path)
        os.remove(self.output_final_path)


    def extract_audio_and_apply_to_video(self):
        """
        Extracts audio from the input video file and applies it to a new video file using FFmpeg.

        """
        # Step 1: Extract audio from the input video
        audio_file = "extracted_audio.aac"  # Temporary file for extracted audio
        extract_audio_command = [
            'ffmpeg', '-i', self.video_path,  # Input video file
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
            'ffmpeg', '-i', self.output_path,  # Input new video file
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
