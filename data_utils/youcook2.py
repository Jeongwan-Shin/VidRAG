import os
import pandas as pd
from glob import glob

class youcook2_dataset():
    def __init__(self):
        # YookCook2 Datasets
        self.video_dir = "../../Llava_train/data/youcook2/raw_videos/training/"
        self.list_label = "../VidRAG/data/youcook2/label_foodtype.csv"
            
        self.data_label = self.load_csv_to_dict(self.list_label)
        self.metadata = self.create_video_metadata(self.data_label, self.video_dir)
        
    def __len__(self):
        return len(self.metadata)
    
    def load_csv_to_dict(self, file_path):
        df = pd.read_csv(file_path, header=None, names=["id", "category"])
        
        return dict(zip(df["id"], df["category"]))

    def create_video_metadata(self, data_dict, base_dir):
        """주어진 ID 값을 기반으로 비디오 파일 경로를 찾아 JSON 구조 생성"""
        video_data = []

        for key, category in data_dict.items():
            video_dir = os.path.join(base_dir, str(key) + "/")  # key 값을 디렉터리 경로로 변환
            video_files = glob(os.path.join(video_dir, "*.mp4"))  # mp4 파일 검색
            
            # TEST : 비디오 하나만 선택하기, 첫 번째 파일 선택 (없으면 None) 
            video_path = video_files[0] if video_files else None  
            if video_path == None:
                continue
                
            video_data.append({
                "id": key,
                "category": category,
                "path": video_path  # 없으면 None이 저장됨
            })
        
        return video_data