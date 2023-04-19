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
model_name = 'jitterflip_seed340_epoch20.pt'
model_name_age = 'onlyAge_jitterflip_seed340_epoch20.pt'
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
    tasks = ["mask", "gender", "age"]

    for i, idx in enumerate(index):
        gt = gts[idx].item()
        pred = preds[idx].item()
        image = np_images[idx]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
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

    
    model_path = os.path.join(args.model_location, model_name) 
    model = load_model(model_path, 'cuda').to('cuda')
    
    model_path_age = os.path.join(args.model_location, model_name_age) 
    model_age = load_model(model_path_age, 'cuda').to('cuda')

    if args.random_seed != -1 : 
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    dataset = MaskBaseDataset(
        data_dir='/opt/ml/input/data/train/images'
    )

    
    dataset_age = AgeDataset(
        data_dir='/opt/ml/input/data/train/images'
    )

    # Data Load
    _, val_set= dataset.split_dataset(val_ratio=0.2, random_seed=args.random_seed)
    _, val_set_age= dataset_age.split_dataset(val_ratio=0.2, random_seed=args.random_seed)

    # Augmentation
    transform = get_transforms()
    val_set.dataset.set_transform(transform['val'])

    transform_age = get_transforms()
    val_set_age.dataset.set_transform(transform_age['val'])

    
    classes = [0 for _ in range(3*2*3)]
    mask_cls = [0 for _ in range(3)]
    gender_cls = [0 for _ in range(2)]
    age_cls = [0 for _ in range(3)]
    age_cls_age = [0 for _ in range(3)]

    ## 전체 val dataset의 클래스별 분류
    for i in range(len(val_set)):
        _, multi_class_label =  val_set[i]
        mask_label, gender_label, age_label = MaskBaseDataset.decode_multi_class(multi_class_label)
        mask_cls[mask_label] +=1
        gender_cls[gender_label] +=1
        age_cls[age_label] +=1
        classes[multi_class_label] += 1

    for i in range(len(val_set_age)):
        _, age_label =  val_set_age[i]
        age_cls_age[age_label] +=1

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader_age = torch.utils.data.DataLoader(
        val_set_age,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    for p in model.parameters():
        p.data = p.data.float()
    for p in model_age.parameters():
        p.data = p.data.float()

    model = model.cuda()
    model_age = model_age.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model,  device_ids=devices)
    model_age = torch.nn.DataParallel(model_age,  device_ids=devices)

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)
    model_parameters_age = [p for p in model_age.parameters() if p.requires_grad]
    optimizer_age = torch.optim.AdamW(model_parameters_age, lr=args.lr, weight_decay=args.wd)

    loss_fn = torch.nn.CrossEntropyLoss()

    correct, count = 0.0, 0.0
    wrong_percent = torch.tensor([0 for _ in range(18)], device='cuda')
    wrong_percent_mask = torch.tensor([0 for _ in range(3)], device='cuda')
    wrong_percent_gender = torch.tensor([0 for _ in range(2)], device='cuda')
    wrong_percent_age = torch.tensor([0 for _ in range(3)], device='cuda')

    # #Evaluate
    test_loader = val_loader
    
    test_loader_age = val_loader_age
    model.eval()
    model_age.eval()

    wrong_imgs = None
    with torch.no_grad():
        print('*'*80)
        print('Starting eval')
        correct, count = 0.0, 0.0
        pbar = tqdm(test_loader)
        pbar_age = tqdm(test_loader_age)
        for i, (batch, batch_age) in enumerate(zip(pbar, pbar_age)):
            batch = maybe_dictionarize_batch(batch)
            batch_age = maybe_dictionarize_batch(batch_age)

            inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)
            inputs_age, labels_age = batch_age['images'].to(DEVICE), batch_age['labels'].to(DEVICE)

            logits = model(inputs)
            logits_age = model_age(inputs_age)


            
            # print('before logit', logits[0])
            # print('before logits_age', logits_age[0])
            # print('before labels', labels[0])

            for i in range(len(logits)): 
                
                age1_percent = 1. + logits_age[i][0] / logits_age[i].sum()
                age2_percent = 1. + logits_age[i][1] / logits_age[i].sum()
                age3_percent = 1. + logits_age[i][2] / logits_age[i].sum()
                
                age_percent = torch.tensor([age1_percent, age2_percent, age3_percent,
                    age1_percent, age2_percent, age3_percent,
                    age1_percent, age2_percent, age3_percent,
                    age1_percent, age2_percent, age3_percent,
                    age1_percent, age2_percent, age3_percent,
                    age1_percent, age2_percent, age3_percent] , device='cuda')

                logits[i] *= age_percent

            loss = loss_fn(logits, labels)
            loss_age = loss_fn(logits_age, labels_age)


            # print('logit', logits[0])
            # print('logits_age', logits_age[0])
            # print('labels', labels[0])


            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

            is_correct = pred.eq(labels.view_as(pred)).squeeze()
            is_wrong = ~is_correct

            wrong_label = labels[is_wrong]
            wrong_pred = pred[is_wrong]

            unique_values, counts = torch.unique(wrong_label, return_counts=True)

            for i in range(len(wrong_label)):
                mask_label, gender_label, age_label = MaskBaseDataset.decode_multi_class(wrong_label[i])
                mask_pred, gender_pred, age_pred = MaskBaseDataset.decode_multi_class(wrong_pred[i])
                if mask_label != mask_pred : 
                    wrong_percent_mask[mask_label] += 1
                if gender_label != gender_pred : 
                    wrong_percent_gender[gender_label] += 1
                if age_label != age_pred : 
                    wrong_percent_age[age_label] += 1

            for value, cnt in zip(unique_values, counts):
                wrong_percent[value] += cnt
            num_true_values_per_col = is_wrong.sum(dim=0)

            count += len(logits)
            pbar.set_description(
                f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}")

            if wrong_imgs is None : 
                wrong_imgs = torch.clone(inputs[is_wrong]).detach().cpu().permute(0, 2, 3, 1).numpy()
                wrong_imgs = MaskBaseDataset.denormalize_image(wrong_imgs, dataset.mean, dataset.std)

            inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
            inputs_np = MaskBaseDataset.denormalize_image(inputs_np, dataset.mean, dataset.std)

            figure = grid_image(inputs_np, labels, pred, is_wrong, n=25) # 16

            dir = f'val_img/{model_name}'
            os.makedirs(dir, exist_ok=True)
            figure.savefig(f'{dir}/my_plot_batch{i}.png')

        top1 = correct / count
        wrong_percent = wrong_percent.to(torch.float)
        classes = torch.tensor(classes, device='cuda')
        mask_cls = torch.tensor(mask_cls, device='cuda')
        gender_cls = torch.tensor(gender_cls, device='cuda')
        age_cls = torch.tensor(age_cls, device='cuda')

        wrong_num = copy.deepcopy(wrong_percent)
        wrong_num_mask = copy.deepcopy(wrong_percent_mask)
        wrong_num_gender = copy.deepcopy(wrong_percent_gender)
        wrong_num_age = copy.deepcopy(wrong_percent_age)

        wrong_percent = wrong_percent/classes * 100
        wrong_percent_mask = wrong_percent_mask/mask_cls * 100
        wrong_percent_gender = wrong_percent_gender/gender_cls * 100
        wrong_percent_age = wrong_percent_age/age_cls * 100

    print('=============')
    for i in range(len(wrong_percent)): 
        print(f'{i} class in total :  {wrong_percent[i]:.4f}, {wrong_num[i]} / {classes[i]}')
    print('-------------')
    for i in range(len(wrong_percent_mask)): 
        print(f'{i} class in mask:  {wrong_percent_mask[i]:.4f}, {wrong_num_mask[i]} / {mask_cls[i]}')
    print('-------------')
    for i in range(len(wrong_percent_gender)): 
        print(f'{i} class in gender:  {wrong_percent_gender[i]:.4f}, {wrong_num_gender[i]} / {gender_cls[i]}')
    print('-------------')
    for i in range(len(wrong_percent_age)): 
        print(f'{i} class in age:  {wrong_percent_age[i]:.4f}, {wrong_num_age[i]} / {age_cls[i]}')

    print('=============')

    print(f'Val acc : {100*top1:.2f}, {correct} / {count}')


