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
video_input_path = "inputvids/rhettandlink_edit3.mp4"
cap = cv2.VideoCapture(video_input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize FaceMesh detector with higher detection confidence
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=2, min_detection_confidence=0.1, min_tracking_confidence=0.5)

# Output video file (without audio)
output_video_path = "outputvids/lip_tracking_video.mp4"

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def detect_lip_movement(frame):
    # Convert BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract upper and lower lip points
            upper_lip_idx = [61, 185, 40, 39, 37, 267, 269, 270, 409]
            lower_lip_idx = [146, 91, 181, 84, 17, 314, 405, 321, 375]

            upper_lip = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in upper_lip_idx])
            lower_lip = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in lower_lip_idx])

            # Convert landmark coordinates to pixel positions
            h, w, _ = frame.shape
            upper_lip_pts = [(int(x * w), int(y * h)) for x, y in upper_lip]
            lower_lip_pts = [(int(x * w), int(y * h)) for x, y in lower_lip]

            # Draw lip landmarks
            for pt in upper_lip_pts + lower_lip_pts:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)  # Green dots for lips

            # Compute lip opening distance
            lip_distance = np.linalg.norm(np.mean(upper_lip, axis=0) - np.mean(lower_lip, axis=0))

            # Extract key landmarks for measuring face size (e.g., width or height)
            left_cheek = face_landmarks.landmark[234]  # Left cheek (Landmark 234)
            right_cheek = face_landmarks.landmark[454]  # Right cheek (Landmark 454)

            # Convert the normalized coordinates to pixel values
            left_cheek = (int(left_cheek.x * w), int(left_cheek.y * h))
            right_cheek = (int(right_cheek.x * w), int(right_cheek.y * h))

            # Calculate the face width (distance between left and right cheeks)
            face_width = calculate_distance(left_cheek, right_cheek)

            # Set the threshold as a proportion of the face width
            threshold = 0.00008 * face_width  # Adjust 0.05 to your needs

            # Display text based on lip distance compared to the face size
            if lip_distance > threshold:
                cv2.putText(frame, "Speaking", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Not Speaking", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Optional: Print lip distance and threshold for debugging
            # print(f"Lip Distance: {lip_distance}, Face Width: {face_width}, Threshold: {threshold}")
    else:
        print("No faces detected")  # Add debugging message if no faces are detected

    return frame

# Get the total number of frames for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize progress bar using tqdm
with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect lip movement and draw landmarks
        frame = detect_lip_movement(frame)

        # Write the frame to the output video
        out.write(frame)

        # Update the progress bar
        pbar.update(1)

# Release video resources
cap.release()
out.release()

# Extract the audio from the original video using ffmpeg
audio_path = "inputvids/rhettandlink_edit2_audio.wav"
ffmpeg.input("inputvids/rhettandlink_edit2.mp4").output(audio_path, ac=1, ar='16000').run(overwrite_output=True)

input_video = ffmpeg.input(output_video_path)
input_audio = ffmpeg.input(audio_path)

# Combine the output video with the original audio using ffmpeg
final_output_path = "outputvids/lip_tracking.mp4"
ffmpeg.concat(input_video, input_audio, v=1, a=1).output(final_output_path, vcodec="h264_nvenc").run()

# Clean up temporary audio file
os.remove(audio_path)
os.remove(output_video_path)

print(f"[✓] Done! Video saved as: {final_output_path}")

