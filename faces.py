import cv2
import mediapipe as mp
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import torch
import os
import math
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disables CUDA for TensorFlow

def generate_id():
    return str(uuid.uuid4())[:8]


def create_face_ids(video_path, max_num_faces, show_video):
    # Setup
    threshold = 0.6  # Cosine similarity threshold for identity matching
    model_name = 'Facenet'  # Can be ArcFace, Facenet512, VGG-Face, etc.
    model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=max_num_faces,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Video input
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames for progress bar

    # Track embeddings + IDs
    face_db = {}  # Stores {face_id: (embedding, bbox)}

    skip_frames = 14

    for i in tqdm(range(math.ceil(total_frames // (skip_frames + 1))), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        h, w, _ = frame.shape
        current_faces = []

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                x_min, y_min, x_max, y_max = w, h, 0, 0
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min, x_max, y_max = min(x_min, x), min(y_min, y), max(x_max, x), max(y_max, y)

                # Padding for better crop
                pad = 50
                x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
                x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

                face_crop = frame[y_min:y_max, x_min:x_max]

                dim = (224, 224)

                face_crop = cv2.resize(face_crop, dim, interpolation=cv2.INTER_AREA)
                face_crop_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float()  # Convert to (C, H, W)
                face_crop_tensor = face_crop_tensor.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')  # Add batch dimension

                # Normalize the face image to the range [-1, 1]
                face_crop_tensor = (face_crop_tensor / 255.0 - 0.5) * 2.0

                try:
                    # Get the embedding for the face crop
                    with torch.no_grad():
                        embedding = model(face_crop_tensor)

                    # Convert to numpy array and append to the list
                    embedding = embedding.cpu().numpy()
                except:
                    print("failed")
                    continue  # Skip if embedding failed

                matched_id = None
                for face_id, (prev_embedding, _, _) in face_db.items():
                    sim = cosine_similarity(embedding.reshape(1, -1), prev_embedding.reshape(1, -1))[0][0]
                    if sim > threshold:
                        matched_id = face_id
                        break

                if matched_id is None:
                    matched_id = generate_id()

                x_avg = (x_max + x_min) / 2
                y_avg = (y_max + y_min) / 2
                if matched_id in face_db:
                    curr_count = face_db[matched_id][1][0]
                    x_avg_count = face_db[matched_id][1][1]
                    y_avg_count = face_db[matched_id][1][2]
                    face_db[matched_id] = (embedding, (curr_count + 1, x_avg_count + x_avg, y_avg_count + y_avg), face_crop)
                else:
                    face_db[matched_id] = (embedding, (1, x_avg, y_avg), face_crop)
                
                # Draw box + ID
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {matched_id}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if show_video:
            cv2.imshow("MediaPipe + DeepFace", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for _ in range(skip_frames):
            ret, frame = cap.read()
            if not ret:
                break

    cap.release()
    cv2.destroyAllWindows()


    print(len(face_db.keys()))
    return face_db
