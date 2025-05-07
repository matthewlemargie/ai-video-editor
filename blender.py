import bpy
import os
import json
import site
import sys
from pathlib import Path
import re

def timecode_to_frame(timecode, fps):
    h, m, s_ms = timecode.split(":")
    s, ms = s_ms.split(",")
    total_seconds = int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
    return int(total_seconds * fps)

def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    entries = content.strip().split("\n\n")
    subtitles = []
    for entry in entries:
        lines = entry.strip().split("\n")
        if len(lines) >= 3:
            time_match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", lines[1])
            if time_match:
                start, end = time_match.groups()
                text = " ".join(lines[2:])
                subtitles.append((start, end, text))
    return subtitles

def add_subtitles_to_vse(srt_path, fps):
    subtitles = parse_srt(srt_path)
    seq = bpy.context.scene.sequence_editor_create()

    for idx, (start_tc, end_tc, text) in enumerate(subtitles):
        start_frame = timecode_to_frame(start_tc, fps) + 1
        end_frame = timecode_to_frame(end_tc, fps) + 1
        duration = end_frame - start_frame

        if end_frame <= start_frame:
            end_frame = start_frame + 1

        text_strip = seq.sequences.new_effect(
            name=f"Sub_{idx}",
            type='TEXT',
            channel=3,
            frame_start=start_frame,
            frame_end=end_frame
        )
        text_strip.text = text
        text_strip.font_size = 56
        text_strip.use_shadow = True
        text_strip.use_outline = True
        text_strip.use_box = True
        text_strip.location = (0.5, 0.3)  # normalized location (x, y)

with open(".last_video.txt", "r") as f:
    video_path = f.readline().strip()
    blend_path = f.readline().strip()
    subtitles_path = f.readline().strip()
    num = int(f.readline().strip())
    den = int(f.readline().strip())
    width = int(f.readline().strip())
    height = int(f.readline().strip())
    
filename = Path(video_path).stem
start_frame = 0
audio_channel = 1
channel = 2

scene = bpy.context.scene

scene.render.fps = num
scene.render.fps_base = den

bpy.context.scene.render.resolution_x = width
bpy.context.scene.render.resolution_y = height

# Create Sequence Editor if it doesn't exist
if not scene.sequence_editor:
    scene.sequence_editor_create()

audio_strip = scene.sequence_editor.sequences.new_sound(
    name="MyAudio",
    filepath=video_path,  # Same file as the video
    channel=audio_channel,
    frame_start=1
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
            frame_start=1
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

add_subtitles_to_vse(subtitles_path, num / den)
