import torch
import cv2
import subprocess
import os
import threading
import sys
import math
import argparse
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

    # Function to play audio with volume control
    def play_segment(self, start, end, volume_var):
        segment = self.audio[int(start * 1000): int(end * 1000)]
        # Save the audio to a temporary file to load in pygame
        temp_filename = "temp_segment.wav"
        segment.export(temp_filename, format="wav")
        
        # Load the audio into pygame
        sound = pygame.mixer.Sound(temp_filename)
        
        # Set the volume using the volume slider value
        volume = volume_var.get()  # Get the current volume value from the slider
        sound.set_volume(volume)  # Set the volume from 0.0 (silent) to 1.0 (full volume)
        
        # Play the sound
        sound.play()
        
        self.is_playing = True  # Set the state to playing

        # Wait until the audio finishes playing
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        
        self.is_playing = False  # Reset the state after playing

    def pause_audio(self):
        if self.is_playing:
            pygame.mixer.pause()
            self.is_playing = False
            pause_button.config(text="Resume")
        else:
            pygame.mixer.unpause()
            self.is_playing = True
            pause_button.config(text="Pause")

    def stop_audio(self):
        pygame.mixer.stop()
        self.is_playing = False
        pause_button.config(text="Pause")

# GUI logic in a thread
    def launch_gui(self, speaker_segments):
        self.root.title("Speaker Segments")

        volume_var = tk.DoubleVar(value=0.75)  # Default volume at 100%

        # Create a volume slider
        volume_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient='horizontal', 
                                 label="Volume Control", variable=volume_var)
        volume_slider.pack(pady=10)

        segments = sorted(speaker_segments, key=lambda x: x[0])

        ids_done = set()

        # Add buttons for each speaker segment
        for i, (id, start, end) in enumerate(segments):
            if id in ids_done:
                continue
            ids_done.add(id)
            label = tk.Label(self.root, text=f"Speaker {id}")
            label.pack(pady=5)
            button = tk.Button(self.root, text=f"{start:.2f}s - {end:.2f}s", 
                               command=lambda s=start, e=end: self.play_segment(s, e, volume_var))
            button.pack(pady=5)

        # Create a pause/resume button
        global pause_button
        pause_button = tk.Button(self.root, text="Pause", command=self.pause_audio)
        pause_button.pack(pady=10)

        # Create a stop button
        stop_button = tk.Button(self.root, text="Stop", command=self.stop_audio)
        stop_button.pack(pady=10)

        # Handle closing gracefully on keyboard interrupt
        def on_closing():
            pygame.mixer.stop()
            self.root.quit()
            sys.exit()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle window close button

        # Main loop to keep GUI running
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            # Capture Ctrl+C and stop the program gracefully
            print("Exiting program...")
            self.root.quit()
            sys.exit()

        return

def show_face_in_tk(face_img):
    # Convert BGR (OpenCV) to RGB
    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    pil_img = Image.fromarray(rgb_img)

    # Convert to Tkinter-compatible image
    tk_img = ImageTk.PhotoImage(pil_img)

    # Create a temporary window to show the image
    win = tk.Toplevel()
    win.title("Face")

    # Keep window on top
    win.attributes("-topmost", True)

    # Display image
    label = tk.Label(win, image=tk_img)
    label.image = tk_img  # Prevent garbage collection
    label.pack()

    return win
