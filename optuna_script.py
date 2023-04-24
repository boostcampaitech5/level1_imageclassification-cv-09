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
from PIL import Image
import torchvision.transforms.functional as TF
import copy
import optuna

from timm.data.transforms_factory import transforms_imagenet_train

from datasets.imagenet import ImageNet98p, ImageNet
from datasets.maskbasedataset import (
    MaskBaseDataset,
    BaseAugmentation,
    get_transforms,
    grid_image,
)
from utils import (
    ModelWrapper,
    maybe_dictionarize_batch,
    cosine_lr,
    get_model_from_sd,
    get_model_from_sd_modified,
)
from zeroshot import zeroshot_classifier
from openai_imagenet_template import openai_imagenet_template
import datasets.maskbasedataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser("model/"),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        help="Model to use -- you can try another like ViT-L/14",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--custom-template",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def objective(trial):
    # init-hyperparameters (optuna parameters)
    hyper_parameters = {
        "epochs": trial.suggest_int("epochs", 10, 25, 1),
        "batch": trial.suggest_categorical("batch", [16, 32, 64, 128, 256, 512]),
        "lr": trial.suggest_categorical("lr", [1e-6, 1e-5, 1e-4, 1e-3]),
        "random_seed": trial.suggest_int("random_seed", 34, 48, 2),
        "i": trial.suggest_int("i", 0, 10, 1),
    }
    print("******************* hyper PARAMETERS~!!!!!!! **********************")
    print(hyper_parameters)
    print("Trial : ", trial.number)
    # wandb.init(
    #     project="optuna",
    #     name=trial.number,
    #     config={
    #         "batch_size": hyper_parameters["batch"],
    #         "lr": hyper_parameters["lr"],
    #         "epochs": hyper_parameters["epochs"],
    #         "i": hyper_parameters["i"],
    #         "random_seed": hyper_parameters["random_seed"],
    #     },
    # )
    # init model & dataset
    class_names = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
    ]
    base_model, preprocess = clip.load(args.model, "cuda", jit=False)
    dataset = MaskBaseDataset(data_dir="/opt/ml/input/data/train/images")
    NUM_CLASSES = len(class_names)
    DEVICE = "cuda"
    if args.custom_template:
        template = [lambda x: f"a photo of a {x}."]
    else:
        template = openai_imagenet_template
    clf = zeroshot_classifier(base_model, class_names, template, DEVICE)

    ######### dataloader load #########
    # Data Load
    train_set, val_set = dataset.split_dataset(
        val_ratio=0.2, random_seed=hyper_parameters["random_seed"]
    )
    train_set.dataset = copy.deepcopy(dataset)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=hyper_parameters["batch"],
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=hyper_parameters["batch"],
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    model_num = hyper_parameters["i"]
    #############모델 load#############
    base_model, preprocess = clip.load("ViT-B/32", "cpu", jit=False)
    model_path = os.path.join(args.model_location, f"model_{model_num}.pt")
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model = get_model_from_sd_modified(
        state_dict, base_model, NUM_CLASSES, initial_weights=clf
    )
    ###################################
    for p in model.parameters():
        p.data = p.data.float()
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    num_batches = len(train_loader)
    optimizer = torch.optim.AdamW(
        model_parameters, lr=hyper_parameters["lr"], weight_decay=args.wd
    )
    scheduler = cosine_lr(
        optimizer,
        hyper_parameters["lr"],
        args.warmup_length,
        hyper_parameters["epochs"] * num_batches,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(hyper_parameters["epochs"]):
        # Train
        correct, count = 0.0, 0.0
        model.train()
        end = time.time()
        for i, batch in enumerate(train_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch["images"].to(DEVICE), batch["labels"].to(DEVICE)
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
                    f"Loss: {loss.item():.6f}\t Acc: {100*correct/count:.2f} \tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",
                    flush=True,
                )
                # wandb.log(
                #     {
                #         "epoch": epoch,
                #         "Train loss": loss.item(),
                #         "Train acc": 100 * correct / count,
                #     }
                # )
                correct, count = 0.0, 0.0

        # Evaluate
        test_loader = val_loader
        model.eval()
        with torch.no_grad():
            print("*" * 80)
            print("Starting eval")
            correct, count = 0.0, 0.0
            pbar = tqdm(test_loader)
            for batch in pbar:
                batch = maybe_dictionarize_batch(batch)
                inputs, labels = batch["images"].to(DEVICE), batch["labels"].to(DEVICE)

                logits = model(inputs)

                loss = loss_fn(logits, labels)

                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                count += len(logits)
                pbar.set_description(
                    f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}"
                )

            top1 = correct / count
        print(f"Val acc at epoch {epoch+1}: {100*top1:.2f}")

        trial.report(loss.item(), epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    # wandb.log(
    #     {
    #         "epoch": epoch + 1,
    #         "Valid loss": loss.item(),
    #         "Valid acc": 100 * top1,
    #     }
    # )
    # wandb.finish()
    return loss.item()


if __name__ == "__main__":
    args = parse_arguments()

    # Optuna study 생성
    study = optuna.create_study(direction="minimize")

    # Optuna study 실행
    study.optimize(objective, n_trials=30)

    # Optuna study 결과 출력
    print(study.best_params)
    print(study.best_value)

    # Optuna 시각화 plot 저장
    """
    TODO : 아래 사진 저장되는 경로 수정
    """
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("/opt/ml/workspace/optuna_images/optuna_history.png")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("/opt/ml/workspace/optuna_images/optuna_param_importances.png")
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image("/opt/ml/workspace/optuna_images/optuna_parallel_coordinates.png")
    fig = optuna.visualization.plot_contour(study)
    fig.write_image("/opt/ml/workspace/optuna_images/optuna_contour.png")
    fig = optuna.visualization.plot_slice(study)
    fig.write_image("/opt/ml/workspace/optuna_images/optuna_slice_plot.png")

    # wandb에 plot 업로드
    wandb.init(project="optuna")
    wandb.log(
        {
            "optuna_history": wandb.Image(
                "/opt/ml/workspace/optuna_images/optuna_history.png"
            )
        }
    )
    wandb.log(
        {
            "optuna_param_importances": wandb.Image(
                "/opt/ml/workspace/optuna_images/optuna_param_importances.png"
            )
        }
    )
    wandb.log(
        {
            "optuna_parallel_coordinates": wandb.Image(
                "/opt/ml/workspace/optuna_images/optuna_parallel_coordinates.png"
            )
        }
    )
    wandb.log(
        {
            "optuna_contour": wandb.Image(
                "/opt/ml/workspace/optuna_images/optuna_contour.png"
            )
        }
    )
    wandb.log(
        {
            "optuna_slice_plot": wandb.Image(
                "/opt/ml/workspace/optuna_images/optuna_slice_plot.png"
            )
        }
    )

    wandb.finish()
