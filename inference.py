import argparse
import torch
import random
from retrival import ClipRetriever
from data_utils import youcook2_dataset

# fixed seed
def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_type", default="youcook2", required=True, help="YooCook2, etc")
    parser.add_argument("--v_ret",  default="openai/clip-vit-base-patch32", required=True, help="Clip, Blip, SigLip, etc") 
    # VLM model arg
    #parser.add_argument("--vlm", required=True, help="Llava-ov-0.5b, Llava-ov-7b, etc")
    
    args = parser.parse_args()
    
    return args
    
def inference():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.data_type == "youcook2":
        print("Loading youcook2")
        videoDB = youcook2_dataset()
        
    retriever = ClipRetriever(model=args.v_ret, device=device, video_metadata=videoDB.metadata)
    
    retriever.incoding_video_process()
    
    text_query = "how to make salmon sashimi?"  # 사용자 입력
    
    top_k_videos = retriever.find_similar_videos(text_query, top_k=5)

    print("\n**Top-K Similar Videos:**")
    for idx, video in enumerate(top_k_videos):
        print(f"{idx+1}. ID: {video['id']} | Category: {video['category']} | Similarity: {video['similarity']:.4f}")
        
if __name__ == "__main__":
    args = get_arguments()
    print("args:", args)
    
    inference()