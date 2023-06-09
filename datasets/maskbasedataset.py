import os
from enum import Enum
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from albumentations import *
from albumentations.pytorch import ToTensorV2
import torch
import random
from typing import Tuple, List
import torchvision
from collections import defaultdict

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")

class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 29:
            return cls.YOUNG
        elif value < 59:
            return cls.MIDDLE
        else:
            return cls.OLD

# gender 병경 함수
def change_sex(gender):
    if gender == 'male':
        return 'female'
    else:
        return 'male'
    
# incorrect <-> normal 변경 함수    
def change_incorrect_normal(file_name):
    if file_name == 'normal':
        return MaskLabels.INCORRECT
    elif file_name == 'incorrect_mask':
        return MaskLabels.NORMAL
    else:
        return MaskLabels.MASK

# incorrect -> mask 변경 함수     
def change_incorrect_to_mask(file_name):
    if file_name == 'incorrect_mask':
        return MaskLabels.MASK
    elif file_name == 'normal':
        return MaskLabels.NORMAL
    else:
        return MaskLabels.MASK

class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }
    
    # relabeling 필요 목록
    relabel_dict = {
        'incorrect_to_from_normal': ['000020', '004418', '005227'],                         # incorrect <-> normal 변경
        'incorrect_to_mask': ['000645', '003574'],                                          # incorrect -> mask 변경
        'incorrect_gender': ['001200', '004432', '005223', '001498-1', '000725',            # gender 변경
                             '006359', '006360', '006361', '006362', '006363', '006364'] 
    }

    def __init__(self,
                data_dir='/opt/ml/input/data/train/images', 
                batch_size=32,
                num_workers=4,
                mean=(0.548, 0.504, 0.479),
                std=(0.237, 0.247, 0.246),
                val_ratio=0.2, 
                random_seed=42):
        
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []

        self.data_dir = data_dir        # '/opt/ml/input/data/train/images' -> 폴더명 ex) 000001_female_Asian_45
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = None
        self.setup()
        self.calc_statistics()
        
        self.populate_test(val_ratio, random_seed)

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                
                if id in self.relabel_dict['incorrect_to_from_normal']:     # incorrect <-> normal 이상치 라벨링 변경
                    mask_label = change_incorrect_normal(_file_name)
                    
                if id in self.relabel_dict['incorrect_to_mask']:            # incorrect -> mask 이상치 라벨링 변경
                    mask_label = change_incorrect_to_mask(_file_name)
                
                if id in self.relabel_dict['incorrect_gender']:             # gender 이상치 라벨링 변경
                    gender = change_sex(gender)
                    
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000] :
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        # assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"
        if self.transform is None :
            print("!!.set_tranform 메소드를 이용하여 transform 을 주입해주세요!!") 
        image = np.array(self.read_image(index))
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        if self.transform is not None : 
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # image_transform = self.transform(image)
        return image, multi_class_label
        

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp
        
    def getSubset(self, indices) -> Subset:
        """
        전체 데이터셋에서 원하는 인덱스 부분만 데이터셋의 서브셋으로 추출
        """
        if indices is None:
            indices = None
        subset = Subset(self, indices)
        return subset

    def split_dataset(self, val_ratio=0.2, random_seed=42) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """

        if val_ratio is None : 
            val_ratio = 1.
            
        n_val = int(len(self) * val_ratio)
        n_train = len(self) - n_val
        if random_seed != -1 : 
            train_set, val_set = random_split(self, [n_train, n_val], generator=torch.Generator().manual_seed(random_seed))
        else : 
            train_set, val_set = random_split(self, [n_train, n_val])

        return train_set, val_set

    def populate_test(self, val_ratio=0.2, random_seed=42):
        transform = get_transforms()
        
        _, val_set = self.split_dataset(val_ratio=val_ratio, random_seed=random_seed)

        val_set.dataset.set_transform(transform['val'])
        # print('test_set[0]', test_set[0])

        self.test_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

class AgeDataset(Dataset):
    num_classes = 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }
    
    # relabeling 필요 목록
    relabel_dict = {
        'incorrect_to_from_normal': ['000020', '004418', '005227'],                         # incorrect <-> normal 변경
        'incorrect_to_mask': ['000645', '003574'],                                          # incorrect -> mask 변경
        'incorrect_gender': ['001200', '004432', '005223', '001498-1', '000725',            # gender 변경
                             '006359', '006360', '006361', '006362', '006363', '006364'] 
    }

    def __init__(self,
                data_dir='/opt/ml/input/data/train/images', 
                batch_size=32,
                num_workers=4,
                mean=(0.548, 0.504, 0.479),
                std=(0.237, 0.247, 0.246),
                val_ratio=0.2, 
                random_seed=42):
        
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []

        self.data_dir = data_dir        # '/opt/ml/input/data/train/images' -> 폴더명 ex) 000001_female_Asian_45
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = None
        self.setup()
        self.calc_statistics()
        
        self.populate_test(val_ratio, random_seed)

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                
                if id in self.relabel_dict['incorrect_to_from_normal']:     # incorrect <-> normal 이상치 라벨링 변경
                    mask_label = change_incorrect_normal(_file_name)
                    
                if id in self.relabel_dict['incorrect_to_mask']:            # incorrect -> mask 이상치 라벨링 변경
                    mask_label = change_incorrect_to_mask(_file_name)
                
                if id in self.relabel_dict['incorrect_gender']:             # gender 이상치 라벨링 변경
                    gender = change_sex(gender)
                    
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        # assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"
        if self.transform is None :
            print("!!.set_tranform 메소드를 이용하여 transform 을 주입해주세요!!") 
        image = np.array(self.read_image(index))
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        if self.transform is not None : 
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # image_transform = self.transform(image)
        return image, age_label
        

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self, val_ratio=0.2, random_seed=42) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """

        if val_ratio is None : 
            val_ratio = 1.
            
        n_val = int(len(self) * val_ratio)
        n_train = len(self) - n_val
        if random_seed != -1 : 
            train_set, val_set = random_split(self, [n_train, n_val], generator=torch.Generator().manual_seed(random_seed))
        else : 
            train_set, val_set = random_split(self, [n_train, n_val])

        return train_set, val_set

    def populate_test(self, val_ratio=0.2, random_seed=42):
        transform = get_transforms()
        
        _, val_set = self.split_dataset(val_ratio=val_ratio, random_seed=random_seed)

        val_set.dataset.set_transform(transform['val'])

        self.test_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

def get_transforms(need=('train', 'train2', 'val'), img_size=(224, 224)):
    """
    train 혹은 validation의 augmentation 함수를 정의합니다. train은 데이터에 많은 변형을 주어야하지만, validation에는 최소한의 전처리만 주어져야합니다.
    
    Args:
        need: 'train', 혹은 'val' 혹은 둘 다에 대한 augmentation 함수를 얻을 건지에 대한 옵션입니다.
        img_size: Augmentation 이후 얻을 이미지 사이즈입니다.
        mean: 이미지를 Normalize할 때 사용될 RGB 평균값입니다.
        std: 이미지를 Normalize할 때 사용될 RGB 표준편차입니다.

    Returns:
        transformations: Augmentation 함수들이 저장된 dictionary 입니다. transformations['train']은 train 데이터에 대한 augmentation 함수가 있습니다.
    """
    mean=(0.56019265, 0.52410305, 0.50145299)
    std=(0.23308824, 0.24294489, 0.2456003)

    transformations = {}
    if 'train' in need:
        transformations['train'] = Compose([
            # CenterCrop(height=412, width=384),
            Resize(img_size[0], img_size[1], p=1.0),
            #Sharpen(p=0.5),
            ColorJitter(p=0.5, hue=0),
            # HorizontalFlip(p=0.5),
            # ShiftScaleRotate(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    if 'train2' in need:        #추가 데이터셋을 위한 증강시 이용하는 transform
        transformations['train2'] = Compose([
            # CenterCrop(height=412, width=384),
            Resize(img_size[0], img_size[1], p=1.0),
            HorizontalFlip(p=1.0),
            # ShiftScaleRotate(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    if 'val' in need:
        transformations['val'] = Compose([
            Resize(img_size[0], img_size[1]),
            #Sharpen(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transformations

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.56019265, 0.52410305, 0.50145299), std=(0.23308824, 0.24294489, 0.2456003)):
        self.img_paths = img_paths
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize, Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

