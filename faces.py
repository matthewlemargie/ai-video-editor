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
import threading

from framediff import is_shot_change

MAX_BUF_LEN = 256

def generate_id():
    return str(uuid.uuid4())[:8]

 
class FaceIDModel:
    def __init__(self, video_path, height=360, embed_threshold=0.7, face_prob_threshold=0.999):
        self.video_path = video_path
        self.embed_threshold = embed_threshold
        self.face_prob_threshold = face_prob_threshold

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Models
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.scale = height / self.h
        self.new_w = int(self.w * self.scale)

        # State
        self.embed_db = {}
        self.position_db = {}
        self.shot_segments = {}
        self.prev = None
        self.last_change_frame = 1
        self.buffer = []

    def send_to_buffer(self, frame_idx):
        ret, frame = self.cap.read()
        if not ret:
            return False
        # Resize and convert
        resized = cv2.resize(frame, (self.new_w, int(self.h * self.scale)))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Shot change
        if self.prev is None:
            shot_change = False
        else:
            shot_change = is_shot_change(self.prev, frame)
        self.prev = frame

        self.buffer.append((shot_change, pil, rgb, frame_idx, frame))
        return True

    def process_buffer(self):
        boxes_list, probs_list = self.mtcnn.detect([x[1] for x in self.buffer])
        for idx, (boxes, probs) in enumerate(zip(boxes_list, probs_list)):
            shot_flag, _, _, frame_idx, orig = self.buffer[idx]
            # Handle shot change
            if shot_flag:
                start = self.last_change_frame
                end   = frame_idx - 1            # close previous shot one frame before this one
                if end >= start:
                    self.shot_segments[(start, end)] = self.position_db.copy()
                self.position_db.clear()
                self.last_change_frame = frame_idx

            if boxes is None:
                continue
            # Filter by probability
            valid = [b for i, b in enumerate(boxes) if probs[i] > self.face_prob_threshold]
            crops, imgs, poses = self._crop_faces(idx, valid)
            if not crops:
                continue
            norm = self._normalize(crops)
            with torch.no_grad():
                embs = self.model(norm).cpu().numpy()

            for j, emb in enumerate(embs):
                fid = self._find_match(emb) or generate_id()
                x1, x2, y1, y2 = poses[j]
                x_avg = (x1 + x2) / 2
                y_avg = (y1 + y2) / 2

                if fid not in self.embed_db:
                    self.embed_db[fid] = (emb, cv2.resize(imgs[j], (112, 112)))

                cnt, xs, ys = self.position_db.get(fid, (0, 0, 0))
                self.position_db[fid] = (cnt + 1, xs + x_avg, ys + y_avg)

        self.buffer.clear()

    def _crop_faces(self, idx, boxes):
        crops, imgs, poses = [], [], []
        shot_flag, pil, rgb, frame_idx, orig = self.buffer[idx]
        for b in boxes:
            x1, y1, x2, y2 = map(int, b)
            pad = 10
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(self.w, x2+pad), min(self.h, y2+pad)
            patch = rgb[y1:y2, x1:x2][:, :, ::-1]
            if patch.size == 0:
                continue
            resized = cv2.resize(patch, (224, 224))
            tensor = torch.from_numpy(resized).permute(2,0,1).float()
            crops.append(tensor)
            imgs.append(patch)
            # scale poses
            poses.append((x1/self.scale, x2/self.scale, y1/self.scale, y2/self.scale))
        return crops, imgs, poses

    def _normalize(self, crops):
        t = torch.stack(crops).to(self.device)
        return (t/255 - 0.5)*2

    def _find_match(self, emb):
        for fid, (ref, _) in self.embed_db.items():
            sim = cosine_similarity(emb.reshape(1,-1), ref.reshape(1,-1))[0][0]
            if sim > self.embed_threshold:
                return fid
        return None

    def run(self):
        for idx in tqdm(range(self.total_frames), desc="Processing frames"):
            alive = self.send_to_buffer(idx+1)
            if not alive:
                break
            if len(self.buffer) >= MAX_BUF_LEN or idx == self.total_frames - 1:
                self.process_buffer()

        # final segment
        self.shot_segments[(self.last_change_frame, self.total_frames)] = self.position_db.copy()
        self.cap.release()
        return self.embed_db, self.shot_segments



class FaceIDModelMultithread:
    def __init__(self, video_path, height=360, embed_threshold=0.7, face_prob_threshold=0.999):
        self.video_path = video_path
        self.embed_threshold = embed_threshold
        self.face_prob_threshold = face_prob_threshold

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Models
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.scale = height / self.h
        self.new_w = int(self.w * self.scale)

        # State
        self.embed_db = {}
        self.position_db = {}
        self.shot_segments = {}
        self.prev = None
        self.last_change_frame = 1
        self._cap_lock = threading.Lock()
        self.buffer = []

    def _package_frame(self, orig_frame, frame_idx):
        # 1) Resize & RGB
        resized = cv2.resize(orig_frame, (self.new_w, int(self.h * self.scale)))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # 2) Shot change
        if self.prev is None:
            shot_change = False
        else:
            shot_change = is_shot_change(self.prev, orig_frame)
        self.prev = orig_frame

        # 3) Return tuple for processing
        return (shot_change, pil, rgb, frame_idx, orig_frame)

    def send_to_buffer(self, frame_idx):
        ret, frame = self.cap.read()
        if not ret:
            return False
        # Resize and convert
        resized = cv2.resize(frame, (self.new_w, int(self.h * self.scale)))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Shot change
        if self.prev is None:
            shot_change = False
        else:
            shot_change = is_shot_change(self.prev, frame)
        self.prev = frame

        self.buffer.append((shot_change, pil, rgb, frame_idx, frame))
        return True

    def process_buffer(self):
        boxes_list, probs_list = self.mtcnn.detect([x[1] for x in self.buffer])
        for idx, (boxes, probs) in enumerate(zip(boxes_list, probs_list)):
            shot_flag, _, _, frame_idx, orig = self.buffer[idx]
            # Handle shot change
            if shot_flag:
                start = self.last_change_frame
                end   = frame_idx - 1
                if end >= start:
                    self.shot_segments[(start, end)] = self.position_db.copy()
                self.position_db.clear()
                self.last_change_frame = frame_idx

            if boxes is None:
                continue
            # Filter by probability
            valid = [b for i, b in enumerate(boxes) if probs[i] > self.face_prob_threshold]
            crops, imgs, poses = self._crop_faces(idx, valid)
            if not crops:
                continue
            norm = self._normalize(crops)
            with torch.no_grad():
                embs = self.model(norm).cpu().numpy()

            for j, emb in enumerate(embs):
                fid = self._find_match(emb) or generate_id()
                x1, x2, y1, y2 = poses[j]
                x_avg = (x1 + x2) / 2
                y_avg = (y1 + y2) / 2

                if fid not in self.embed_db:
                    self.embed_db[fid] = (emb, cv2.resize(imgs[j], (112, 112)))

                cnt, xs, ys = self.position_db.get(fid, (0, 0, 0))
                self.position_db[fid] = (cnt + 1, xs + x_avg, ys + y_avg)

        self.buffer.clear()

    def _crop_faces(self, idx, boxes):
        crops, imgs, poses = [], [], []
        shot_flag, pil, rgb, frame_idx, orig = self.buffer[idx]
        for b in boxes:
            x1, y1, x2, y2 = map(int, b)
            pad = 10
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(self.w, x2+pad), min(self.h, y2+pad)
            patch = rgb[y1:y2, x1:x2][:, :, ::-1]
            if patch.size == 0:
                continue
            resized = cv2.resize(patch, (224, 224))
            tensor = torch.from_numpy(resized).permute(2,0,1).float()
            crops.append(tensor)
            imgs.append(patch)
            poses.append((x1/self.scale, x2/self.scale, y1/self.scale, y2/self.scale))
        return crops, imgs, poses

    def _normalize(self, crops):
        t = torch.stack(crops).to(self.device)
        return (t/255 - 0.5)*2

    def _find_match(self, emb):
        for fid, (ref, _) in self.embed_db.items():
            sim = cosine_similarity(emb.reshape(1,-1), ref.reshape(1,-1))[0][0]
            if sim > self.embed_threshold:
                return fid
        return None

    def _fill_buffer(self, buffer, start_idx):
        buffer.clear()
        idx = start_idx
        for _ in range(MAX_BUF_LEN):
            with self._cap_lock:
                ret, frame = self.cap.read()
            if not ret:
                break
            buffer.append(self._package_frame(frame, idx))
            idx += 1
        return idx

    def _fill_buffer_threaded(self, start_idx):
        def worker():
            self._next_end_idx = self._fill_buffer(self._next_buffer, start_idx)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread

    def _swap_buffers(self):
        self.buffer, self._next_buffer = self._next_buffer, self.buffer
        self.current_start_idx = self.current_end_idx

    def run(self):
        self.buffer = []
        self._next_buffer = []
        self.current_start_idx = 1
        self.current_end_idx = 1

        fill_thread = self._fill_buffer_threaded(self.current_start_idx)

        while True:
            fill_thread.join()
            if not self.buffer:
                break

            self.process_buffer()
            self.current_end_idx = getattr(self, "_next_end_idx", self.current_start_idx)

            if self.current_end_idx > self.total_frames:
                break

            fill_thread = self._fill_buffer_threaded(self.current_end_idx)
            self._swap_buffers()

        # final shot segment
        self.shot_segments[(self.last_change_frame, self.total_frames)] = self.position_db.copy()
        self.cap.release()
        return self.embed_db, self.shot_segments

