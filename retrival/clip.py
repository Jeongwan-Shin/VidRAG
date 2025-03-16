import os
import pandas as pd
from glob import glob
import av
import numpy as np
from tqdm import tqdm 
import json 
from PIL import Image

from transformers import CLIPProcessor, CLIPModel
import torch

class ClipRetriever():
    def __init__(self, model, device, metadata):
        # CLIP Based Retrival Model
        self.clip_model = CLIPModel.from_pretrained(model).to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(model)
        self.device = device
        self.metadata = metadata
        self.FEATURE_DIR = "../data/yoocook2/features"
        self.JSON_PATH = "../data/yoocook2/video_metadata.json"

    def read_video_frames(self, video_path, fps):
        """비디오에서 1초당 1개 프레임을 추출하여 PIL.Image 리스트로 반환."""
        frames = []
        container = av.open(video_path)

        indices = np.arange(0, container.streams.video[0].duration, fps)

        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                img = Image.fromarray(frame.to_ndarray(format="rgb24"))  # PIL 이미지 변환
                frames.append(img)

        return frames

    def encode_frames_with_clip(self, frames):
        """CLIP의 Vision Encoder로 프레임을 인코딩하여 특징 벡터 반환."""
        if not frames:
            return None  # 프레임이 없으면 None 반환

        inputs = self.clip_processor(images=frames, return_tensors="pt").to(self.device)  # CUDA 이동
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)  # Vision 인코딩
        return features.cpu().numpy()  # GPU -> CPU 변환 후 NumPy 배열 반환

    def save_features(feature_array, feature_path):
        """인코딩된 feature를 npy 파일로 저장"""
        np.save(feature_path, feature_array)
    
    def incoding_video_process(self, video_metadata):
        """각 비디오를 읽고, CLIP Vision Encoder로 인코딩 후, feature를 개별 파일로 저장."""
        for entry in tqdm(video_metadata, desc="Processing Videos", unit="video"):
            video_id = entry["id"]
            category = entry["category"]
            paths = entry["path"]

            entry["feature_files"] = []  # feature 파일 경로 저장

            if not paths or len(paths) == 0:
                continue

            if not os.path.exists(paths):
                print(f"File not found: {paths}")
                continue
            
            feature_path = os.path.join(self.FEATURE_DIR, f"{video_id}_{os.path.basename(paths)}.npy")

            # 이미 feature 파일이 존재하면 스킵
            if os.path.exists(feature_path):
                print(f"Feature file already exists, skipping: {feature_path}")
                entry["feature_files"].append(feature_path)  # JSON에 feature 경로 추가
                continue

            if not os.path.exists(paths):
                print(f"File not found: {paths}")
                continue

            # 비디오 프레임 읽기
            container = av.open(paths)
            fps = int(container.streams.video[0].average_rate)  # FPS 가져오기
            frames = self.read_video_frames(paths, fps)

            # CLIP 인코딩 수행
            encoded_frames = self.encode_frames_with_clip(frames)
            if encoded_frames is not None:
                #feature_path = os.path.join(self.FEATURE_DIR, f"{video_id}_{os.path.basename(paths)}.npy")
                self.save_features(encoded_frames, feature_path)  # 파일 저장
                entry["feature_files"].append(feature_path)  # JSON에 feature 경로 추가

            print(f"Processed Video {video_id}: {paths}, Frames: {len(frames)}, Saved Feature: {feature_path}")
            
        print(f"Save Video metadata : Path: {self.JSON_PATH}")
        with open(self.JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)
    
    def load_video_metadata(self):
        """JSON 파일에서 비디오 메타데이터를 로드"""
        with open(self.JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def encode_text_query(self, text_query):
        """CLIP의 Text Encoder를 사용하여 텍스트 쿼리를 인코딩"""
        inputs = self.clip_processor(text=[text_query], return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)  # 정규화

    def load_features(feature_path):
        """저장된 feature 파일을 불러와서 PyTorch Tensor로 변환"""
        feature_array = np.load(feature_path)
        return torch.tensor(feature_array)

    def find_similar_videos(self, text_query, top_k=5):
        """입력된 텍스트 쿼리와 가장 유사한 비디오를 찾음."""
        video_metadata = self.load_video_metadata()  
        text_embedding = self.encode_text_query(text_query)

        similarities = []  # 유사도 리스트

        for entry in tqdm(video_metadata, desc="Computing Similarities", unit="video"):
            video_id = entry["id"]
            category = entry["category"]
            feature_files = entry.get("feature_files", [])

            if not feature_files:
                continue  # feature 파일이 없는 경우 스킵

            frame_similarities = []  # 각 프레임별 유사도를 저장할 리스트

            for feature_path in feature_files:
                if os.path.exists(feature_path):
                    video_features = self.load_features(feature_path)

                    for feature in video_features:
                        frame_sim = torch.nn.functional.cosine_similarity(text_embedding, feature.to(self.device))
                        frame_similarities.extend(frame_sim.tolist())  # 리스트로 변환하여 저장
                    
                    # frame_sim = torch.nn.functional.cosine_similarity(text_embedding, video_features.to(device))
                    # frame_similarities.extend(frame_sim.tolist())  # 리스트로 변환하여 저장

            if not frame_similarities:
                continue  # 유사도 계산된 프레임이 없으면 스킵

            avg_similarity = sum(frame_similarities) / len(frame_similarities)  # 프레임별 유사도의 평균 계산
            #print(video_id, feature_path, avg_similarity)
            similarities.append({"id": video_id, "category": category, "similarity": avg_similarity})

        # 유사도 순으로 정렬하여 Top-K 반환
        similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:top_k]

        return similarities