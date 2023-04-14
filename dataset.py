from torch.utils.data import Dataset
import os


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)

    def __len__(self):
        # 데이터셋의 총 데이터 수 반환
        for data in self.image_files:
            print(data)
        return len(self.image_files)

    def __getitem__(self, idx):
        # idx 번째 데이터 반환
        # 예시) return self.data[idx]
        return self.image_files[idx]
