from typing import Any, Callable, Optional, Union
from my_dataset import MyDataset
from pprint import pprint
import os
import shutil
import argparse

from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torchmetrics.detection import MeanAveragePrecision

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
)
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.transforms.v2 import Compose, ToTensor, Resize


def fasterrcnn_resnet101_base(
        trainable_backbone_layers: Optional[int] = None,
        weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V1
):
    weights_backbone = ResNet101_Weights.verify(weights_backbone)

    backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

    is_trained = weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(
        is_trained, trainable_backbone_layers, 5, 3
    )
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)


    model.backbone = backbone

    return model



def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--lr", "-l", type=float, default=1e-4)
    parser.add_argument(
        "--data_path", "-d", type=str, default="dataset/ElementosCircuitoElectrico_2"
    )
    parser.add_argument("--num_epochs", "-n", type=int, default=50)
    parser.add_argument("--save_path", "-c", type=str, default="trained_models")
    parser.add_argument("--log_path", "-t", type=str, default="tensorboard")
    parser.add_argument("--checkpoint_path", "-p", type=str, default=None)

    args = parser.parse_args()
    return args


def collate_fn(batch):
    images = []
    targets = []
    for image, taget in batch:
        images.append(image)
        targets.append(taget)
    return images, targets


def train(args):
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    writer = SummaryWriter(args.log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([ToTensor()])

    train_dataset = MyDataset(args.data_path, True, transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_dataset = MyDataset(args.data_path, False, transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    num_classes = len(train_dataset.class_name)
    num_iters = len(train_dataloader)

    model = fasterrcnn_resnet101_base()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, num_classes * 4)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["best_map"]
        model.load_state_dict(checkpoint["model_params"])
    else:
        best_map = -1
        start_epoch = 0
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="BLUE")
        for iter, (images, targets) in enumerate(progress_bar):

            images = [image.to(device) for image in images]
            target_device = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            if len(target_device[0]["boxes"]) == 0:
                continue

            losses = model(images, target_device)
            loss = sum([loss for loss in losses.values()])
            progress_bar.set_description(
                "Epoch: {}/{}. Loss: {:0.4f}".format(epoch + 1, args.num_epochs, loss)
            )

            writer.add_scalar("Train/Loss", loss.item(), epoch * num_iters + iter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()
        model.eval()
        progress_bar = tqdm(val_dataloader, colour="YELLOW")
        metric = MeanAveragePrecision(iou_type="bbox")

        for images, targets in progress_bar:
            if len(target_device[0]["boxes"]) == 0:
                continue
            images = [image.to(device) for image in images]
            with torch.no_grad():
                predicts = model(images)

            predic_to_cpu = [
                {key: value.to("cpu") for key, value in predict.items()}
                for predict in predicts
            ]
            metric.update(predic_to_cpu, targets)
            del predicts, images
        map = metric.compute()
        pprint(map)
        checkpoint = {
            "epoch": epoch + 1,
            "best_map": best_map,
            "model_params": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last.pt"))
        if best_map < map["map"]:
            torch.save(checkpoint, os.path.join(args.save_path, "best.pt"))
            best_map = map["map"]

        writer.add_scalar("Val/mAP", map["map"], epoch)
        writer.add_scalar("Val/mAP_50", map["map_50"], epoch)
        writer.add_scalar("Val/mAP_75", map["map_75"], epoch)


if __name__ == "__main__":
    args = get_args()
    train(args)

