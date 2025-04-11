import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from deepface.DeepFace import build_model
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import os
import threading
from time import time
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disables CUDA for TensorFlow

face_db_mutex = threading.Lock()
face_id_counter_mutex = threading.Lock()

def generate_id():
    return str(uuid.uuid4())[:8]

def process_frames(video, face_db, example_faces, model_name, model, threshold, max_num_faces):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=max_num_faces,
                                       min_detection_confidence=0.1, min_tracking_confidence=0.5)
    for i in tqdm(range(len(video)), desc="Processing frames", unit="frame"):
        rgb_frame = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        h, w, _ = video[i].shape
        current_faces = []

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                x_min, y_min, x_max, y_max = w, h, 0, 0
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min, x_max, y_max = min(x_min, x), min(y_min, y), max(x_max, x), max(y_max, y)

                # Padding for better crop
                pad = 75
                x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
                x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

                face_crop = video[i][y_min:y_max, x_min:x_max]

                scale_percent = 75  # Resize to 50% of original size
                width = int(face_crop.shape[1] * scale_percent / 100)
                height = int(face_crop.shape[0] * scale_percent / 100)
                dim = (width, height)

                face_crop = cv2.resize(face_crop, dim, interpolation=cv2.INTER_AREA)

                try:
                    # Get embedding from DeepFace
                    t = time()
                    embedding_obj = DeepFace.represent(img_path=face_crop, model_name=model_name, model=model, enforce_detection=False)
                    print(time() - t)
                    embedding = embedding_obj[0]['embedding']
                except:
                    print("failed")
                    continue  # Skip if embedding failed

                matched_id = None
                with face_db_mutex:
                    for face_id, (prev_embedding, _) in face_db.items():
                        sim = cosine_similarity([embedding], [prev_embedding])[0][0]
                        if sim > threshold:
                            matched_id = face_id
                            break

                if matched_id is None:
                    matched_id = generate_id()

                x_avg = (x_max + x_min) / 2
                y_avg = (y_max + y_min) / 2
                with face_db_mutex:
                # Save to database
                    if matched_id in face_db:
                        curr_count = face_db[matched_id][1][0]
                        x_avg_count = face_db[matched_id][1][1]
                        y_avg_count = face_db[matched_id][1][2]
                        face_db[matched_id] = (embedding, (curr_count + 1, x_avg_count + x_avg, y_avg_count + y_avg))
                    else:
                        face_db[matched_id] = (embedding, (1, x_avg, y_avg))
                        example_faces[matched_id] = face_crop
                
                # Draw box + ID
                cv2.rectangle(video[i], (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(video[i], f"ID: {matched_id}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("MediaPipe + DeepFace", video[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def create_face_ids(video_path, max_num_faces):
    # Setup
    threshold = 0.5  # Cosine similarity threshold for identity matching
    model_name = 'Facenet'  # Can be ArcFace, Facenet512, VGG-Face, etc.
    model = build_model(model_name)
    from PIL import Image

    img = np.ones((160,160,3), dtype=np.uint8) * 255  # White square
    embedding_obj = DeepFace.represent(img_path=img, model_name=model_name, enforce_detection=False, detector_backend='mediapipe')
    if embedding_obj:
        print(embedding_obj)
    else:
        print("dudredrdurdurd")

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=max_num_faces,
                                       min_detection_confidence=0.1, min_tracking_confidence=0.5)

    # Video input
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames for progress bar

    # Track embeddings + IDs
    face_id_counter = 0
    face_db = {}  # Stores {face_id: (embedding, bbox)}
    example_faces = {}

    #Read in all frames
    video = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video.append(frame) 
    cap.release()

    process_frames(video, face_db, example_faces, model_name, model, threshold, max_num_faces)

    cv2.destroyAllWindows()

    for k, v in face_db.items():
        face_db[k] = v[1]

    return face_db, example_faces


    # num_threads = 4
    # active_threads = []

    # clips = np.array_split(video, num_threads)

    # for clip in clips:
        # while len(active_threads) >= num_threads:
            # for t in active_threads:
                # if not t.is_alive():  # Check if the thread is done
                    # t.join()  # Ensure it's fully terminated
                    # cv2.destroyAllWindows()
                    # active_threads.remove(t)  # Remove from active list
        # # Start a new thread
        # t = threading.Thread(target=process_frames, args=(clip, face_db, example_faces, model_name, threshold))
        # t.start()
        # active_threads.append(t)

    # Ensure all threads finish before exiting
    # for t in active_threads:
        # t.join()

