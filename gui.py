import torch
import cv2
import os
import sys
import tkinter as tk
import tkinter.simpledialog as sd
from PIL import Image, ImageTk
import simpleaudio as sa
import pygame
import time
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

class GUI:
    def __init__(self, video_path):
        self.root = tk.Tk()
        pygame.mixer.init()
        self.audio = AudioSegment.from_file(video_path)
        self.audio_data = pygame.mixer.Sound(self.audio.raw_data)
        # Variable to keep track of whether the audio is playing
        self.is_playing = False
        self.faces_to_speakers = {}


    # Function to play audio with volume control
    def play_segment(self, start, end, volume_var):
        pygame.mixer.stop()
        segment = self.audio[int(start * 1000): int(end * 1000)]
        # Save the audio to a temporary file to load in pygame
        temp_filename = "temp_segment.wav"
        segment.export(temp_filename, format="wav")
        
        # Load the audio into pygame
        sound = pygame.mixer.Sound(temp_filename)

        os.remove(temp_filename)
        
        # Set the volume using the volume slider value
        volume = volume_var.get()  # Get the current volume value from the slider
        sound.set_volume(volume)  # Set the volume from 0.0 (silent) to 1.0 (full volume)
        
        # Play the sound
        sound.play()

        
    def stop_audio(self):
        pygame.mixer.stop()


    def match_faces_to_voices(self, face_ids, example_faces, speaker_segments):
        self.root.title("Speakers to Faces")

        # Frame for showing face pics and entry boxes
        faces_frame = tk.Frame(self.root, padx=10, pady=10)
        faces_frame.pack(padx=10, pady=10)

        self.entry_boxes = {}     # To keep track of entries per face_id

        for face_id in face_ids:
            rgb_img = cv2.cvtColor(example_faces[face_id], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            tk_img = ImageTk.PhotoImage(pil_img)

            # Column for image + entry box
            face_column = tk.Frame(faces_frame)
            face_column.pack(side="right", padx=10)

            img_label = tk.Label(face_column, image=tk_img)
            img_label.image = tk_img
            img_label.pack()

            entry = tk.Entry(face_column)
            entry.pack()
            self.entry_boxes[face_id] = entry

        audios_frame = tk.Frame(self.root, padx=10, pady=10)
        audios_frame.pack(padx=20, pady=20)

        volume_var = tk.DoubleVar(value=0.75)  # Default volume at 100%

        # Create a volume slider
        volume_slider = tk.Scale(audios_frame, from_=0, to=1, resolution=0.01, orient='horizontal', 
                                 label="Volume Control", variable=volume_var)
        volume_slider.pack(pady=10)

        segments = sorted(speaker_segments, key=lambda x: x[0])

        ids_done = set()

        # Add buttons for each speaker segment
        for i, (id, start, end) in enumerate(segments):
            if id in ids_done:
                continue
            ids_done.add(id)
            label = tk.Label(audios_frame, text=f"Speaker {id}")
            label.pack(pady=5)
            button = tk.Button(audios_frame, text=f"{start:.2f}s - {end:.2f}s", 
                               command=lambda s=start, e=end: self.play_segment(s, e, volume_var))
            button.pack(pady=5)

        # Create a stop button
        stop_button = tk.Button(audios_frame, text="Stop", command=self.stop_audio)
        stop_button.pack(pady=10)

        # Handle closing gracefully on keyboard interrupt
        def on_closing():
            pygame.mixer.stop()
            self.root.quit()
            sys.exit()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle window close button

        # Submit button at the bottom
        submit_btn = tk.Button(self.root, text="Submit", command=self.on_submit)
        submit_btn.pack(pady=10)

        # Main loop to keep GUI running
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            # Capture Ctrl+C and stop the program gracefully
            print("Exiting program...")
            self.root.quit()
            sys.exit()


    def on_submit(self):
        for face_id, entry in self.entry_boxes.items():
            value = entry.get().strip()
            if value.isdigit():
                self.faces_to_speakers[face_id] = int(value)
        self.root.destroy()
