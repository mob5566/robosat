import os
import sys
import argparse
import collections
import time
from contextlib import contextmanager

from PIL import Image

import torch
import torchvision
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, CenterCrop, Normalize
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50, lraspp_mobilenet_v3_large

from tqdm import tqdm

from robosat.transforms import (
    JointCompose,
    JointTransform,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    ConvertImageMode,
    ImageToTensor,
    MaskToTensor,
)
from robosat.datasets import SlippyMapTilesConcatenation
from robosat.metrics import Metrics
from robosat.losses import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
from robosat.unet import UNet
from robosat.vanilla_unet import VanillaUNet
from robosat.deeplabv3 import DeepLabV3P
from robosat.utils import plot
from robosat.config import load_config
from robosat.log import Log


@contextmanager
def no_grad():
    with torch.no_grad():
        yield


def add_parser(subparser):
    parser = subparser.add_parser(
        "train", help="trains model on dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model", type=str, required=True, help="path to model configuration file")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("--checkpoint", type=str, required=False, help="path to a model checkpoint (to retrain)")
    parser.add_argument("--resume", type=bool, default=False, help="resume training or fine-tuning (if checkpoint)")
    parser.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")
    parser.add_argument("--epochs", type=int, help="number of epochs for training")
    parser.add_argument("--lr", type=float, help="learning rate for training")
    parser.add_argument("--loss", type=str, help="loss function for training")
    parser.add_argument("--testdir", type=str, help="path to testing dataset")

    parser.set_defaults(func=main)


def main(args):
    model = load_config(args.model)
    dataset = load_config(args.dataset)

    if args.epochs:
        model["opt"]["epochs"] = args.epochs
    if args.lr:
        model["opt"]["lr"] = args.lr
    if args.loss:
        model["opt"]["loss"] = args.loss

    output_dir = os.path.join(
        model["common"]["checkpoint"],
        "{model}_{epochs:05d}_{lr}_{loss}_{dataset}_{date}".format(**{
            "model": model["common"]["model"],
            "epochs": model["opt"]["epochs"],
            "lr": model["opt"]["lr"],
            "loss": model["opt"]["loss"],
            "dataset": dataset["common"]["name"],
            "date": time.strftime("%Y-%m-%d-%H%M%S")
        })
    )

    device = torch.device("cuda" if model["common"]["cuda"] else "cpu")

    if model["common"]["cuda"] and not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")

    os.makedirs(output_dir, exist_ok=True)
    writer_train = SummaryWriter(log_dir=os.path.join(output_dir, "train"))
    writer_val = SummaryWriter(log_dir=os.path.join(output_dir, "valid"))
    if args.testdir:
        writer_test = SummaryWriter(log_dir=os.path.join(output_dir, "test"))

    num_classes = len(dataset["common"]["classes"])
    if "model" not in model["common"] or model["common"]["model"] == "unet":
        net = UNet(num_classes)
    elif model["common"]["model"] == "vanilla_unet":
        net = VanillaUNet(n_channels=3, n_classes=num_classes)
    elif model["common"]["model"] == "fcn":
        net = fcn_resnet50(num_classes=num_classes)
    elif model["common"]["model"] == "deeplabv3":
        net = deeplabv3_resnet50(num_classes=num_classes)
    elif model["common"]["model"] == "lraspp":
        net = lraspp_mobilenet_v3_large(num_classes=num_classes)
    elif model["common"]["model"] == "deeplabv3p":
        net = DeepLabV3P(num_classes=num_classes)
    net = DataParallel(net)
    net = net.to(device)

    if model["common"]["cuda"]:
        torch.backends.cudnn.benchmark = True

    try:
        weight = torch.Tensor(dataset["weights"]["values"])
    except KeyError:
        if model["opt"]["loss"] in ("CrossEntropy", "mIoU", "Focal"):
            sys.exit("Error: The loss function used, need dataset weights values")

    optimizer = Adam(net.parameters(), lr=model["opt"]["lr"])

    resume = 0
    if args.checkpoint:

        def map_location(storage, _):
            return storage.cuda() if model["common"]["cuda"] else storage.cpu()

        # https://github.com/pytorch/pytorch/issues/7178
        chkpt = torch.load(args.checkpoint, map_location=map_location)
        net.load_state_dict(chkpt["state_dict"])

        if args.resume:
            optimizer.load_state_dict(chkpt["optimizer"])
            resume = chkpt["epoch"]

    if model["opt"]["loss"] == "CrossEntropy":
        criterion = CrossEntropyLoss2d(weight=weight).to(device)
    elif model["opt"]["loss"] == "mIoU":
        criterion = mIoULoss2d(weight=weight).to(device)
    elif model["opt"]["loss"] == "Focal":
        criterion = FocalLoss2d(weight=weight).to(device)
    elif model["opt"]["loss"] == "Lovasz":
        criterion = LovaszLoss2d().to(device)
    else:
        sys.exit("Error: Unknown [opt][loss] value !")

    train_loader, val_loader = get_dataset_loaders(model, dataset, args.workers)

    test_loader = None
    if args.testdir:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        target_size = (model["common"]["image_size"],) * 2
        batch_size = model["common"]["batch_size"]

        test_transform = JointCompose(
            [
                JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
                JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        )

        test_dataset = SlippyMapTilesConcatenation(
            [os.path.join(args.testdir, "images")],
            os.path.join(args.testdir, "labels"),
            test_transform
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
            num_workers=args.workers
        )

    num_epochs = model["opt"]["epochs"]
    if resume >= num_epochs:
        sys.exit("Error: Epoch {} set in {} already reached by the checkpoint provided".format(num_epochs, args.model))

    history = collections.defaultdict(list)
    log = Log(os.path.join(output_dir, "log"))

    log.log("--- Hyper Parameters on Dataset: {} ---".format(dataset["common"]["dataset"]))
    log.log("Batch Size:\t {}".format(model["common"]["batch_size"]))
    log.log("Image Size:\t {}".format(model["common"]["image_size"]))
    log.log("Learning Rate:\t {}".format(model["opt"]["lr"]))
    log.log("Loss function:\t {}".format(model["opt"]["loss"]))
    log.log("Test Dir:\t {}".format(args.testdir))
    if "weight" in locals():
        log.log("Weights :\t {}".format(dataset["weights"]["values"]))
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.log('Number of parameters: {}'.format(pytorch_total_params))
    log.log("---")

    for epoch in range(resume, num_epochs):
        log.log("Epoch: {}/{}".format(epoch + 1, num_epochs))

        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)
        log.log(
            "Train    loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                train_hist["loss"],
                train_hist["miou"],
                dataset["common"]["classes"][1],
                train_hist["fg_iou"],
                train_hist["mcc"],
            )
        )
        writer_train.add_scalar("Loss", train_hist["loss"], epoch)
        writer_train.add_scalar("mIoU", train_hist["miou"], epoch)
        writer_train.add_scalar("FG_IoU", train_hist["fg_iou"], epoch)
        writer_train.add_scalar("MCC", train_hist["mcc"], epoch)

        for k, v in train_hist.items():
            history["train " + k].append(v)

        val_hist = validate(val_loader, num_classes, device, net, criterion)
        log.log(
            "Validate loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                val_hist["loss"], val_hist["miou"], dataset["common"]["classes"][1], val_hist["fg_iou"], val_hist["mcc"]
            )
        )
        writer_val.add_scalar("Loss", val_hist["loss"], epoch)
        writer_val.add_scalar("mIoU", val_hist["miou"], epoch)
        writer_val.add_scalar("FG_IoU", val_hist["fg_iou"], epoch)
        writer_val.add_scalar("MCC", val_hist["mcc"], epoch)

        for k, v in val_hist.items():
            history["val " + k].append(v)

        if (epoch + 1) % 25 == 0:
            visual = "history-{:05d}-of-{:05d}.png".format(epoch + 1, num_epochs)
            plot(os.path.join(output_dir, visual), history)

            checkpoint = "checkpoint-{:05d}-of-{:05d}.pth".format(epoch + 1, num_epochs)

            states = {"epoch": epoch + 1, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()}

            torch.save(states, os.path.join(output_dir, checkpoint))

        if test_loader:
            test_hist = validate(test_loader, num_classes, device, net, criterion)
            log.log(
                "Test loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                    test_hist["loss"], test_hist["miou"],
                    dataset["common"]["classes"][1], test_hist["fg_iou"],
                    test_hist["mcc"]
                )
            )
            writer_test.add_scalar("Loss", test_hist["loss"], epoch)
            writer_test.add_scalar("mIoU", test_hist["miou"], epoch)
            writer_test.add_scalar("FG_IoU", test_hist["fg_iou"], epoch)
            writer_test.add_scalar("MCC", test_hist["mcc"], epoch)


def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.train()

    for images, masks in tqdm(loader, desc="Train", unit="batch", ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        optimizer.zero_grad()
        outputs = net(images)["out"]

        assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }


@no_grad()
def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.eval()

    for images, masks in tqdm(loader, desc="Validate", unit="batch", ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        outputs = net(images)["out"]

        assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks)

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            metrics.add(mask, output)

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }


def get_dataset_loaders(model, dataset, workers):
    target_size = (model["common"]["image_size"],) * 2
    batch_size = model["common"]["batch_size"]
    path = dataset["common"]["dataset"]
    training_set = []
    validation_set = []

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointRandomHorizontalFlip(0.5),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )

    valid_transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )

    if ("other_data" not in dataset["common"] or
        len(dataset["common"]["other_data"]) == 0):
        training_set.append(os.path.join(path, "training", "images"))
        validation_set.append(os.path.join(path, "validation", "images"))
    else:
        for other_data in dataset["common"]["other_data"]:
            training_set.append(os.path.join(path, "training", other_data))
            validation_set.append(os.path.join(path, "validation", other_data))

    train_dataset = SlippyMapTilesConcatenation(
        training_set, os.path.join(path, "training", "labels"), train_transform
    )

    val_dataset = SlippyMapTilesConcatenation(
        validation_set, os.path.join(path, "validation", "labels"), valid_transform
    )

    assert len(train_dataset) > 0, "at least one tile in training dataset"
    assert len(val_dataset) > 0, "at least one tile in validation dataset"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)

    return train_loader, val_loader
