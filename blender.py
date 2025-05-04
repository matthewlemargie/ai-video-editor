import bpy
import os
import json
import site
import sys
from pathlib import Path

conda_site_packages = "/opt/anaconda/envs/editor/lib/python3.10/site-packages"

# Add to sys.path if not already present
if conda_site_packages not in sys.path:
    sys.path.append(conda_site_packages)
    site.addsitedir(conda_site_packages)

import ffmpeg
input_dir = "/home/dmac/Documents/projects2025/ai-video-editor/inputvids/"
video_file = "vip_edit4.mp4"
 
# === CONFIGURATION ===
video_path = input_dir + video_file
filename = Path(video_path).stem
blend_path = f"/home/dmac/Documents/projects2025/ai-video-editor/cache/{filename}_blend.json"  # <-- Replace with your actual file
start_frame = 0
audio_channel = 1
channel = 2


probe = ffmpeg.probe(video_path)
video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
r_frame_rate = video_stream['r_frame_rate']  # e.g., "30000/1001"

num, den = map(int, r_frame_rate.split('/'))

scene = bpy.context.scene

scene.render.fps = num
scene.render.fps_base = den

bpy.context.scene.render.resolution_x = 608
bpy.context.scene.render.resolution_y = 1080

# Create Sequence Editor if it doesn't exist
if not scene.sequence_editor:
    scene.sequence_editor_create()

audio_strip = scene.sequence_editor.sequences.new_sound(
    name="MyAudio",
    filepath=video_path,  # Same file as the video
    channel=audio_channel,
    frame_start=start_frame
)

audio_strip.volume  = 0.6

with open(blend_path, "r") as f:
    blend = json.load(f)
    
prev_left, prev_right = blend[0][3], blend[0][4]
start = 1

for frame_idx, bottom, top, left, right in blend:
    
    if left != prev_left and right != prev_right or frame_idx == len(blend):
        end = frame_idx

        movie_strip = scene.sequence_editor.sequences.new_movie(
            name="MyVideo",
            filepath=video_path, 
            channel=channel,
            frame_start=0
        )
        movie_strip.frame_offset_start = start - 1
        if frame_idx != len(blend):
            movie_strip.frame_final_duration = end - start

        offset = -((prev_left + prev_right) // 2 - 1920 / 2)
        
        movie_strip.transform.offset_x = offset
        
        prev_left = left
        prev_right = right
        start = frame_idx

bpy.context.scene.frame_end = end - 1

# Remove the video strip but keep the audio
# audio_strip = movie_strip.sound  # Access the audio that comes with the movie strip
# audio_strip.channel = channel
