import cv2
import subprocess
import math
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import numpy as np

from gui import GUI
from diarize import diarize
from faces import create_face_ids
from subtitles import generate_word_srt, add_subtitles_from_srt 


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class TikTokEditor:
    def __init__(self, video_path, n_speakers, max_num_faces, show_video):
        self.n_speakers = n_speakers
        self.max_num_faces = max_num_faces
        self.show_video = show_video
        self.video_path = video_path
        self.video_title = Path(video_path).stem
        self.output_path = os.path.join("output", "output.mp4")
        self.output_final_path = os.path.join("output", "output_final.mp4")
        self.output_final_subtitled_path = os.path.join("output", f"{self.video_title}_final_subtitled.mp4")
        self.segments_path = os.path.join("segments_cache", f"{Path(self.video_path).stem}_segments.json")
        self.subtitle_path = os.path.join("subtitles_cache", f"{self.video_title}.srt")
        self.ids_dict = {}


    def analyze(self):
        if os.path.exists(self.segments_path):
            with open(self.segments_path, "r") as f:
                self.speaker_segments = json.load(f)
        else:
            self.speaker_segments = diarize(self.video_path, n_speakers=self.n_speakers)
            with open(self.segments_path, "w") as f:
                json.dump(self.speaker_segments, f, indent=4, default=convert_to_serializable)
        self.face_db, self.example_faces = create_face_ids(self.video_path, max_num_faces=self.max_num_faces, show_video=self.show_video)
        self.face_ids = self.face_db.keys()

        self.gui = GUI(self.video_path)
        self.gui.match_faces_to_voices(self.face_ids, self.example_faces, self.speaker_segments)
        self.combine_speakers_faces()


    def edit(self):
        self.crop_video_on_speaker_bbox_static()
        self.extract_audio_and_apply_to_video()


    def edit_w_subtitles(self):
        self.crop_video_on_speaker_bbox_static()
        self.extract_audio_and_apply_to_video()
        self.create_subtitle_video()


    def combine_speakers_faces(self):
        to_delete = set()
        for face_id, speaker_id in self.gui.faces_to_speakers.items():
            if speaker_id not in self.ids_dict:
                self.ids_dict[speaker_id] = face_id
            else:
                main_face_id = self.ids_dict[speaker_id]
                main_face_info = self.face_db[main_face_id]
                new_face_info = self.face_db[face_id]
                self.face_db[main_face_id] = tuple(x + y for x, y in zip(main_face_info, new_face_info))
                to_delete.add(face_id)

        for id in to_delete:
            del self.face_db[id]

        for k, v in self.face_db.items():
            avgs = (int(self.face_db[k][1]/self.face_db[k][0]), int(self.face_db[k][2]/self.face_db[k][0]))
            self.face_db[k] = avgs


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

        boxes = {}
        for k, v in self.face_db.items():
            box = (v[0] - new_width//2, v[0] + new_width//2)
            boxes[k] = box

        frame_index = 0
        current_speaker = None
        timeline_index = 0

        # Editing loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = frame_index / fps

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

            # Get the face ID for the current speaker
            face_id = self.ids_dict.get(current_speaker)
            bbox = boxes.get(face_id)

            if bbox:
                x1, x2 = bbox
            else:
                x1, x2 = 0, new_width

            # Handle cases where faces are too close to edge
            if x2 > width:
                cropped = frame[0:height, -new_width:]
            elif x1 < 0:
                cropped = frame[0:height, 0:new_width]
            else:
                cropped = frame[0:height, x1:x2]

            out.write(cropped)
            frame_index += 1

        cap.release()
        out.release()
        print("Done writing:", self.output_path)


    def create_subtitle_video(self):
        generate_word_srt(self.output_final_path, self.subtitle_path)
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
