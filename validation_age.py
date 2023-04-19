import argparse
import os
import torch
import clip
import os
from tqdm import tqdm
import time
import wandb
import random 
import numpy as np
import matplotlib.pyplot as plt

from timm.data.transforms_factory import transforms_imagenet_train

from datasets.imagenet import ImageNet98p, ImageNet
from datasets.maskbasedataset import MaskBaseDataset, BaseAugmentation, get_transforms, AgeDataset
from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr, get_model_from_sd
from zeroshot import zeroshot_classifier
import torchvision.transforms.functional as TF
import copy


#############입력하세요#############
model_name = 'onlyAge0_epoch15.pt'
###################################

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('model/'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5, ## 0.00002
    )
    parser.add_argument(
        "--name",
        default='finetune_cp',
        help='Filename for the checkpoints.'
    )
    parser.add_argument(
        "--random-seed", 
        type=int,
        default=42,
    )
    parser.add_argument(
        "--i",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
    )

    return parser.parse_args()

def grid_image(np_images, gts, preds, is_wrong, n=16):
    batch_size = np_images.shape[0]
    index = is_wrong.nonzero()

    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["age"]

    for i, idx in enumerate(index):
        gt = gts[idx].item()
        pred = preds[idx].item()
        image = np_images[idx]
        title = "\n".join([
            f"{tasks[0]} - gt: {gt}, pred: {pred}"
        ])

        plt.subplot(n_grid, n_grid, i + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def load_model(model_path, device):
    base_model, preprocess = clip.load('ViT-B/32', 'cuda', jit=False)
    print('model_path', model_path)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model = get_model_from_sd(state_dict, base_model)

    return model

if __name__ == '__main__':
    
    args = parse_arguments()
    DEVICE = 'cuda'

    NUM_CLASSES = 3
    
    model_path = os.path.join(args.model_location, model_name) 
    model = load_model(model_path, 'cuda').to('cuda')

    if args.random_seed != -1 : 
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    dataset = AgeDataset(
        data_dir='/opt/ml/input/data/train/images'
    )

    # Data Load
    _, val_set= dataset.split_dataset(val_ratio=0.2, random_seed=args.random_seed)

    # Augmentation
    transform = get_transforms()
    val_set.dataset.set_transform(transform['val'])

    age_cls = [0 for _ in range(3)]

    ## 전체 val dataset의 클래스별 분류
    # for i in range(len(val_set)):
    #     _, multi_class_label =  val_set[i]
    #     mask_label, gender_label, age_label = MaskBaseDataset.decode_multi_class(multi_class_label)
    #     mask_cls[mask_label] +=1
    #     gender_cls[gender_label] +=1
    #     age_cls[age_label] +=1
    #     classes[multi_class_label] += 1

    for i in range(len(val_set)):
        _, age_label =  val_set[i]
        age_cls[age_label] +=1

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )


    for p in model.parameters():
        p.data = p.data.float()

    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model,  device_ids=devices)

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)

    loss_fn = torch.nn.CrossEntropyLoss()

    correct, count = 0.0, 0.0
    wrong_percent_age = torch.tensor([0 for _ in range(3)], device='cuda')

    # #Evaluate
    test_loader = val_loader
    model.eval()

    wrong_imgs = None
    with torch.no_grad():
        print('*'*80)
        print('Starting eval')
        correct, count = 0.0, 0.0
        pbar = tqdm(test_loader)
        for i, batch in enumerate(pbar):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)

            logits = model(inputs)

            loss = loss_fn(logits, labels)

            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

            is_correct = pred.eq(labels.view_as(pred)).squeeze()
            is_wrong = ~is_correct

            wrong_label = labels[is_wrong]
            wrong_pred = pred[is_wrong]


            for i in range(len(wrong_label)):
                wrong_percent_age[wrong_label[i]] += 1

            count += len(logits)
            pbar.set_description(
                f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}")

            inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
            inputs_np = MaskBaseDataset.denormalize_image(inputs_np, dataset.mean, dataset.std)

            figure = grid_image(inputs_np, labels, pred, is_wrong, n=25) # 16

            dir = f'val_img/{model_name}'
            os.makedirs(dir, exist_ok=True)
            figure.savefig(f'{dir}/my_plot_batch{i}.png')

        top1 = correct / count
        age_cls = torch.tensor(age_cls, device='cuda')
        wrong_num_age = copy.deepcopy(wrong_percent_age)
        wrong_percent_age = wrong_percent_age/age_cls * 100

    print('=============')
    for i in range(len(wrong_percent_age)): 
        print(f'{i} class in age:  {wrong_percent_age[i]:.4f}, {wrong_num_age[i]} / {age_cls[i]}')

    print('=============')

    print(f'Val acc : {100*top1:.2f}, {correct} / {count}')


