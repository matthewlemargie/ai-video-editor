import cv2
import mediapipe as mp
import numpy as np
import ffmpeg
from tqdm import tqdm
import os

# Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# Open video
video_input_path = "inputvids/kylejesse_edit1.mp4"
cap = cv2.VideoCapture(video_input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize FaceMesh detector
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=2, min_detection_confidence=0.1, min_tracking_confidence=0.5)

# Output video file (without audio)
output_video_path = "outputvids/temp_tracking_video.mp4"

# Placeholder to determine output resolution after first frame
_, sample_frame = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to frame 0
crop_height = int(frame_height * 0.7)
crop_width = int(crop_height * 9 / 16)
output_size = (crop_width, crop_height)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def detect_lip_movement(frame):
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    cropped_faces = []
    speaking_flags = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Lip landmarks
            upper_idx = [61, 185, 40, 39, 37, 267, 269, 270, 409]
            lower_idx = [146, 91, 181, 84, 17, 314, 405, 321, 375]
            upper = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in upper_idx])
            lower = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in lower_idx])
            lip_dist = np.linalg.norm(np.mean(upper, axis=0) - np.mean(lower, axis=0))

            left = face_landmarks.landmark[234]
            right = face_landmarks.landmark[454]
            face_width = calculate_distance((left.x * w, left.y * h), (right.x * w, right.y * h))
            threshold = 0.00008 * face_width
            speaking = lip_dist > threshold
            speaking_flags.append(speaking)

            # Face centroid
            all_points = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
            centroid = np.mean(all_points, axis=0).astype(int)
            cx, cy = centroid

            x1 = max(0, cx - crop_width // 2)
            y1 = max(0, cy - crop_height // 2)
            x2 = min(w, x1 + crop_width)
            y2 = min(h, y1 + crop_height)

            x1 = max(0, x2 - crop_width)
            y1 = max(0, y2 - crop_height)

            cropped = frame[y1:y2, x1:x2]
            cropped_faces.append(cropped)

    return frame, cropped_faces, speaking_flags

# Track the last speaker index
last_speaker_idx = -1

# Progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        full_frame, cropped_faces, speaking_flags = detect_lip_movement(frame)

        # Determine active speaker
        active_speaker_idx = -1
        for idx, is_speaking in enumerate(speaking_flags):
            if is_speaking:
                active_speaker_idx = idx
                break

        if active_speaker_idx == -1:
            active_speaker_idx = last_speaker_idx

        if 0 <= active_speaker_idx < len(cropped_faces):
            cropped = cropped_faces[active_speaker_idx]
            cropped_resized = cv2.resize(cropped, output_size)
            out.write(cropped_resized)
            last_speaker_idx = active_speaker_idx
        else:
            black_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            out.write(black_frame)

        pbar.update(1)

# Cleanup
cap.release()
out.release()

# Extract and attach audio
audio_path = "inputvids/temp.wav"
ffmpeg.input(video_input_path).output(audio_path, ac=1, ar='16000').run(overwrite_output=True)
input_video = ffmpeg.input(output_video_path)
input_audio = ffmpeg.input(audio_path)
final_output_path = f"outputvids/{video_input_path.split('/')[-1].split('.')[0]}_lip_tracking.mp4"
ffmpeg.concat(input_video, input_audio, v=1, a=1).output(final_output_path, vcodec="h264_nvenc").run()

# Remove temp files
os.remove(audio_path)
os.remove(output_video_path)
print(f"[✓] Done! Video saved as: {final_output_path}")

