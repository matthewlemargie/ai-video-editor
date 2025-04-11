from diarize import diarize
from faces import create_face_ids

import torch
import tensorflow as tf
import cv2
import subprocess
import os
import math

def check_torch_cuda():
    if torch.cuda.is_available():
        print("🚀 CUDA/GPU is being used for processing.")
    else:
        print("⚡ CUDA/GPU is not available, running on CPU.")

def check_tf_cuda():
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

check_torch_cuda()
check_tf_cuda()

video_path = "inputvids/kylejesse_edit1.mp4"

# Diarize audio
speaker_segments = diarize(video_path, n_speakers=2)

# Assign ids to faces in video
face_db, example_faces = create_face_ids(video_path, max_num_faces=2)

for k, v in face_db.items():
    print("ID: ", k)

face_ids = face_db.keys()

print(speaker_segments)

ids_dict = {}

to_delete = set()
for face_id in face_ids:
    cv2.imshow(" ", example_faces[face_id])
    cv2.waitKey(1)
    while True:
        try:
            speaker_id = input(f"Which speaker belongs to face ID {face_id}: ")
            if speaker_id:
                speaker_id = int(speaker_id)
                if speaker_id not in ids_dict:
                    ids_dict[speaker_id] = face_id
                else:
                    main_face_id = ids_dict[speaker_id]
                    main_face_info = face_db[main_face_id]
                    new_face_info = face_db[face_id]
                    face_db[main_face_id] = tuple (x + y for x, y in zip(main_face_info, new_face_info))
                    to_delete.add(face_id)
            cv2.destroyAllWindows()
            break
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            pass

for id in to_delete:
    del face_db[id]
            
for k, v in face_db.items():
    avgs = (int(face_db[k][1]/face_db[k][0]), int(face_db[k][2]/face_db[k][0]))
    face_db[k] = avgs


def crop_video_on_speaker_bbox_static(
    video_path,
    output_path,
    face_positions,
    speaker_timeline,
    speaker_to_face,
):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate new width for 9:16 aspect ratio
    new_width = int(math.ceil(height * (9 / 16)))
    output_size = (new_width, height)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    boxes = {}
    for k, v in face_positions.items():
        box = (v[0] - new_width//2, v[0] + new_width//2)
        boxes[k] = box

    frame_index = 0
    current_speaker = None
    timeline_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_index / fps

        # Determine current speaker
        while timeline_index < len(speaker_timeline):
            speaker_id, start, end = speaker_timeline[timeline_index]
            if start <= time_sec <= end:
                current_speaker = speaker_id
                break
            elif time_sec > end:
                timeline_index += 1
            else:
                break

        # Get the face ID for the current speaker
        face_id = speaker_to_face.get(current_speaker)
        bbox = boxes.get(face_id)

        if bbox:
            x1, x2 = bbox
        else:
            x1, x2 = 0, 608

        cropped = frame[0:height, x1:x2]
        out.write(cropped)
        frame_index += 1

    cap.release()
    out.release()
    print("Done writing:", output_path)


def extract_audio_and_apply_to_video(input_video_file, new_video_file, output_file):
    """
    Extracts audio from the input video file and applies it to a new video file using FFmpeg.

    Args:
    - input_video_file (str): Path to the input video file (from which to extract audio).
    - new_video_file (str): Path to the new video file (to which audio should be applied).
    - output_file (str): Path where the final video with the extracted audio will be saved.
    """
    # Step 1: Extract audio from the input video
    audio_file = "extracted_audio.aac"  # Temporary file for extracted audio
    extract_audio_command = [
        'ffmpeg', '-i', input_video_file,  # Input video file
        '-vn',  # No video stream
        '-acodec', 'aac',  # Audio codec (AAC)
        '-strict', 'experimental',  # Allow experimental codecs
        audio_file  # Output audio file
    ]
    
    # Run the extract audio command
    subprocess.run(extract_audio_command, check=True)
    print(f"Audio extracted to {audio_file}")
    
    # Step 2: Apply the extracted audio to the new video
    apply_audio_command = [
        'ffmpeg', '-i', new_video_file,  # Input new video file
        '-i', audio_file,  # Input extracted audio file
        '-c:v', 'copy',  # Copy video stream (no re-encoding)
        '-c:a', 'aac',  # Encode audio as AAC
        '-strict', 'experimental',  # Allow experimental codecs
        '-map', '0:v:0',  # Map the first video stream
        '-map', '1:a:0',  # Map the first audio stream
        output_file  # Output file path
    ]
    
    # Run the apply audio command
    subprocess.run(apply_audio_command, check=True)
    print(f"Final video with audio saved to {output_file}")
    
    # Optional: Remove the temporary audio file after processing
    subprocess.run(['rm', audio_file])
    print(f"Temporary audio file {audio_file} removed.")

output_path = "outputvids/output.mp4"
crop_video_on_speaker_bbox_static(video_path, output_path, face_db, speaker_segments, ids_dict)
extract_audio_and_apply_to_video(video_path, output_path, "outputvids/output_final.mp4")
