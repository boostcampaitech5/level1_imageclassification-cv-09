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

from timm.data.transforms_factory import transforms_imagenet_train

from datasets.imagenet import ImageNet98p, ImageNet
from datasets.maskbasedataset import MaskBaseDataset, BaseAugmentation, get_transforms, grid_image
from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr, get_model_from_sd
from zeroshot import zeroshot_classifier
from openai_imagenet_template import openai_imagenet_template

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
        "--custom-template", action="store_true", default=False,
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
        "--wd",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--model",
        default='ViT-B/32',
        help='Model to use -- you can try another like ViT-L/14'
    )
    parser.add_argument(
        "--name",
        default='finetune_cp',
        help='Filename for the checkpoints.'
    )
    parser.add_argument(
        "--timm-aug", action="store_true", default=False,
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

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    DEVICE = 'cuda'

    if args.random_seed != -1 : 
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    if args.custom_template:
        template = [lambda x : f"a photo of a {x}."]
    else:
        template = openai_imagenet_template

    base_model, preprocess = clip.load(args.model, 'cuda', jit=False)
    
    # 98p is the 98% of ImageNet train set that we train on -- the other 2% is hodl-out val.
    if args.timm_aug:
        train_preprocess = transforms_imagenet_train(
                img_size=base_model.visual.input_resolution,
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
    else:
        train_preprocess = preprocess

    # train_dset = ImageNet98p(train_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    # test_dset = ImageNet(preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)

    dataset = MaskBaseDataset(
        data_dir='/opt/ml/input/data/train/images'
    )
    num_classes = dataset.num_classes  # 18

    # Data Load
    train_set, val_set= dataset.split_dataset(val_ratio=0.2, random_seed=args.random_seed)
    # print("train_set[0]", train_set[0])
    # print("val_set[0]", val_set[0])


    # Augmentation
    transform = get_transforms()
    train_set.dataset.set_transform(transform['train'])
    val_set.dataset.set_transform(transform['val'])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    class_names = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen']
    clf = zeroshot_classifier(base_model, class_names, template, DEVICE)
    NUM_CLASSES = dataset.num_classes  
    feature_dim = base_model.visual.output_dim

    # state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # model = ModelWrapper(base_model, feature_dim, NUM_CLASSES, normalize=True, initial_weights=clf)


    #############모델 load#############
    base_model, preprocess = clip.load('ViT-B/32', 'cuda', jit=False)
    model_path = os.path.join(args.model_location, f'model_{args.i}.pt') 
    state_dict = torch.load(model_path, map_location=torch.device('cuda'))
    model = get_model_from_sd(state_dict, base_model)
    ###################################


    for p in model.parameters():
        p.data = p.data.float()

    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model,  device_ids=devices)

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)


    num_batches = len(train_loader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    loss_fn = torch.nn.CrossEntropyLoss()

    wandb.init(name=args.name+str(args.i), config={"batch_size": args.batch_size,
                    "lr"        : args.lr,
                    "epochs"    : args.epochs,
                    "name"      : args.name,
                    "criterion_name" : loss_fn})

    for epoch in range(args.epochs):
        # Train
        correct, count = 0.0, 0.0
        model.train()
        end = time.time()
        for i, batch in enumerate(train_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)
            data_time = time.time() - end

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            count += len(logits)
            if i % 20 == 0:
                
                percent_complete = 100.0 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\t Acc: {100*correct/count:.2f} \tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

                wandb.log({
                    "epoch" : epoch,
                    "Train loss": loss.item(),
                    "Train acc" : 100*correct/count
                })
                correct, count = 0.0, 0.0

        # #Evaluate
        test_loader = val_loader
        model.eval()
        with torch.no_grad():
            print('*'*80)
            print('Starting eval')
            correct, count = 0.0, 0.0
            pbar = tqdm(test_loader)
            figure = None
            for batch in pbar:
                batch = maybe_dictionarize_batch(batch)
                inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)

                logits = model(inputs)

                loss = loss_fn(logits, labels)

                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                count += len(logits)
                pbar.set_description(
                    f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}")
                
                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = MaskBaseDataset.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(inputs_np, labels, pred, n=25, shuffle=True) # 16
            top1 = correct / count
        print(f'Val acc at epoch {epoch+1}: {100*top1:.2f}')

        if (epoch+1) % 5 == 0 :
            model_path = os.path.join(args.model_location, f'{args.name}{args.i}_epoch{epoch+1}.pt')
            print('Saving model to', model_path)
            torch.save(model.module.state_dict(), model_path)

        wandb.log({
            "epoch" : epoch+1,
            "Valid loss": loss.item(),
            "Valid acc" : 100*top1,
            "Valid fig" : wandb.Image(figure)
        })

    wandb.finish()