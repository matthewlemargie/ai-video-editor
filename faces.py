import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import torch
import os
import math
from tqdm import tqdm
import time

from framediff import is_shot_change

# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disables CUDA for TensorFlow

def generate_id():
    return str(uuid.uuid4())[:8]

def create_face_ids_mtcnn(video_path, max_num_faces, show_video):
    # Setup
    embed_threshold = 0.7  # Cosine similarity threshold for identity matching
    face_prob_threshold = 0.999
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

    prev = None
    last_change_frame = 1
    buffer = []
    MAX_BUF_LEN = 256

    for i in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        height = 360
        h, w = frame.shape[:2]
        scale = height / h
        new_w = int(w * scale)
        resized_frame = cv2.resize(frame, (new_w, height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        if prev is not None:
            shot_change = is_shot_change(prev, frame)
        else:
            shot_change= False

        prev = frame

        buffer.append((shot_change, pil_frame, rgb_frame))

        if len(buffer) == MAX_BUF_LEN:
            boxes_list, probs_list = mtcnn.detect([x[1] for x in buffer])

            batch = list(zip(boxes_list, probs_list))

            for batch_idx, (boxes, probs) in enumerate(batch):
                buffer_shot_change = buffer[batch_idx][0]
                if buffer_shot_change:
                    shot_segments[(last_change_frame, i - (MAX_BUF_LEN - batch_idx) + 1)] = position_db.copy()
                    position_db = {}
                    last_change_frame = i - (MAX_BUF_LEN - batch_idx) + 1 + 1

                if boxes is None or len(boxes) == 0:
                    continue

                face_boxes = []
                for j, box in enumerate(boxes):
                    if probs[j] > face_prob_threshold:
                        face_boxes.append(box)

                boxes = face_boxes

                face_crops = []
                faces = []
                faces_poses = []

                if face_boxes is not None:
                    for box in face_boxes:
                        x1, y1, x2, y2 = box
                        # x1, y1, x2, y2 = 1/scale * x1, 1/scale * y1, 1/scale * x2, 1/scale * y2 

                        # Padding for better crop
                        pad = 10

                        x1 = max(0, min(w - 1, x1 - pad))
                        y1 = max(0, min(h - 1, y1 - pad))
                        x2 = max(0, min(w, x2 + pad))
                        y2 = max(0, min(h, y2 + pad))

                        frame_for_box = buffer[batch_idx][2][:, :, ::-1]
                        face_crop = frame_for_box[int(y1):int(y2), int(x1):int(x2)]

                        if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                            continue  # skip this face box

                        dim = (224, 224)
                        face_crop = cv2.resize(face_crop, dim, interpolation=cv2.INTER_AREA)
                        faces.append(face_crop)
                        face_crop_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float()  # Convert to (C, H, W)
                        face_crops.append(face_crop_tensor)
                        x1, y1, x2, y2 = 1/scale * x1, 1/scale * y1, 1/scale * x2, 1/scale * y2 
                        faces_poses.append((x1, x2, y1, y2))

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
                            if sim > embed_threshold:
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
                            cv2.putText(frame, f"ID: {matched_id}", (faces_poses[j][0], faces_poses[j][1] - 10),
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

