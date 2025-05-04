import cv2
import mediapipe as mp
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import torch
import os
import math
from tqdm import tqdm
from time import time

from framediff import is_shot_change

# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disables CUDA for TensorFlow

def generate_id():
    return str(uuid.uuid4())[:8]

def create_face_ids_mtcnn(video_path, max_num_faces, show_video):
    # Setup
    threshold = 0.6  # Cosine similarity threshold for identity matching
    # Face embedding model
    model_name = 'Facenet'
    model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    # Face detection model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=device)

    # Video input
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Track embeddings + IDs
    embed_db = {}
    position_db = {}
    shot_segments = {}

    skip_frames = 0

    prev = None

    last_change_frame = 1

    buffer = []

    for i in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        height = 720
        h, w = frame.shape[:2]
        scale = height / h
        new_w = int(w * scale)
        resized_frame = cv2.resize(frame, (new_w, height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        buffer.append(pil_frame)

        if prev is not None:
            shot_change = is_shot_change(prev, frame)
        else:
            shot_change= False

        prev = frame


        if len(buffer) == 128 or shot_change:
            if shot_change:
                shot_segments[(last_change_frame, i)] = position_db.copy()
                position_db = {}
                last_change_frame = i + 1
            boxes_list, probs_list = mtcnn.detect(buffer)
            # boxes, probs = mtcnn.detect(rgb_frame)

            batch = list(zip(boxes_list, probs_list))

            for boxes, probs in batch:

                face_boxes = []
                for j, box in enumerate(boxes):
                    if probs[j] > 0.7:
                        face_boxes.append(box)

                boxes = face_boxes

                face_crops = []
                faces = []
                faces_poses = []

                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = 1/scale * x1, 1/scale * y1, 1/scale * x2, 1/scale * y2 

                        # Padding for better crop
                        pad = 50

                        x1 = max(0, int(x1 - pad))
                        y1 = max(0, int(y1 - pad))
                        x2 = min(w, int(x2 + pad))
                        y2 = min(h, int(y2 + pad))

                        face_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                        dim = (224, 224)

                        face_crop = cv2.resize(face_crop, dim, interpolation=cv2.INTER_AREA)
                        faces.append(face_crop)
                        face_crop_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float()  # Convert to (C, H, W)
                        face_crops.append(face_crop_tensor)
                        faces_poses.append((x1, x2, y1, y2))
                        # Draw box + ID
                        if show_video:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {matched_id}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


                face_crops = np.array(face_crops)
                if face_crops.size > 0:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    face_crop_tensor = torch.from_numpy(face_crops).to(device)

                    # Normalize the face images to [-1, 1]
                    face_crop_tensor = (face_crop_tensor / 255.0 - 0.5) * 2.0

                    try:
                        with torch.no_grad():
                            embeddings = model(face_crop_tensor)
                        embeddings = embeddings.cpu().numpy()  # Convert embeddings to NumPy for similarity comparison
                    except :
                        print("Embedding failed")
                        continue

                    for j, embedding in enumerate(embeddings):
                        matched_id = None
                        for face_id, (prev_embedding, _) in embed_db.items():
                            # Ensure both embeddings are numpy arrays
                            sim = cosine_similarity(embedding.reshape(1, -1), prev_embedding.reshape(1, -1))[0][0]
                            if sim > threshold:
                                matched_id = face_id
                                break

                        if matched_id is None:
                            matched_id = generate_id()  # Make sure `generate_id()` is implemented

                        x_avg = (faces_poses[j][1] + faces_poses[j][0]) / 2
                        y_avg = (faces_poses[j][3] + faces_poses[j][2]) / 2

                        if matched_id not in embed_db:
                            embed_db[matched_id] = (embedding, cv2.resize(faces[j], (112, 112), interpolation=cv2.INTER_AREA))

                        if matched_id not in position_db:
                            position_db[matched_id] = (1, x_avg, y_avg)
                        else:
                            curr_count = position_db[matched_id][0]
                            x_avg_count = position_db[matched_id][1]
                            y_avg_count = position_db[matched_id][2]
                            position_db[matched_id] = (curr_count + 1, x_avg_count + x_avg, y_avg_count + y_avg)

                        # Draw box + ID
                        if show_video:
                            cv2.rectangle(frame, (faces_poses[j][0], faces_poses[j][2]), (faces_poses[j][1], faces_poses[j][3]), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {matched_id}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            buffer.clear()
        if show_video:
            cv2.imshow("MediaPipe + DeepFace", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if position_db is not {}:
        shot_segments[(last_change_frame, total_frames)] = position_db.copy()

    cap.release()
    cv2.destroyAllWindows()

    return embed_db, shot_segments


def create_face_ids(video_path, max_num_faces, show_video):
    # Setup
    threshold = 0.6  # Cosine similarity threshold for identity matching
    model_name = 'Facenet'
    model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=max_num_faces,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Video input
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Track embeddings + IDs
    embed_db = {}
    position_db = {}
    shot_segments = {}

    skip_frames = 0

    prev = None

    last_change_frame = 1

    for i in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_start = time()
        result = face_mesh.process(rgb_frame)
        print("mesh process time")
        print(time() - mesh_start)

        h, w, _ = frame.shape
        current_faces = []

        if prev is not None:
            shot_change = is_shot_change(prev, frame)
        else:
            shot_change= False

        prev = frame

        if shot_change:
            shot_segments[(last_change_frame, i)] = position_db.copy()
            position_db = {}
            last_change_frame = i + 1

        face_crops = []
        faces = []
        faces_poses = []

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
                faces.append(face_crop)
                face_crop_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float()  # Convert to (C, H, W)
                face_crops.append(face_crop_tensor)
                faces_poses.append((x_min, x_max, y_min, y_max))
                # Draw box + ID
                if show_video:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {matched_id}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        face_crops = np.array(face_crops)
        if face_crops.size > 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            face_crop_tensor = torch.from_numpy(face_crops).to(device)

            # Normalize the face images to [-1, 1]
            face_crop_tensor = (face_crop_tensor / 255.0 - 0.5) * 2.0

            try:
                with torch.no_grad():
                    embeddings = model(face_crop_tensor)
                embeddings = embeddings.cpu().numpy()  # Convert embeddings to NumPy for similarity comparison
            except :
                print("Embedding failed")
                continue

            for i, embedding in enumerate(embeddings):
                matched_id = None
                for face_id, (prev_embedding, _) in embed_db.items():
                    # Ensure both embeddings are numpy arrays
                    sim = cosine_similarity(embedding.reshape(1, -1), prev_embedding.reshape(1, -1))[0][0]
                    if sim > threshold:
                        matched_id = face_id
                        break

                if matched_id is None:
                    matched_id = generate_id()  # Make sure `generate_id()` is implemented

                x_avg = (faces_poses[i][1] + faces_poses[i][0]) / 2
                y_avg = (faces_poses[i][3] + faces_poses[i][2]) / 2

                if matched_id not in embed_db:
                    embed_db[matched_id] = (embedding, cv2.resize(faces[i], (112, 112), interpolation=cv2.INTER_AREA))

                if matched_id not in position_db:
                    position_db[matched_id] = (1, x_avg, y_avg)
                else:
                    curr_count = position_db[matched_id][0]
                    x_avg_count = position_db[matched_id][1]
                    y_avg_count = position_db[matched_id][2]
                    position_db[matched_id] = (curr_count + 1, x_avg_count + x_avg, y_avg_count + y_avg)

                # Draw box + ID
                if show_video:
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

    if position_db is not {}:
        shot_segments[(last_change_frame, total_frames)] = position_db.copy()

    cap.release()
    cv2.destroyAllWindows()

    return embed_db, shot_segments
