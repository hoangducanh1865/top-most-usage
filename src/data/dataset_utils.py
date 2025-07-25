import topmost
from src.config.config import DEVICE

def get_dataset():
    topmost.download_dataset("20NG", cache_path="./datasets")
    dataset = topmost.BasicDataset("./datasets/20NG", device=DEVICE, read_labels=True)
    return dataset