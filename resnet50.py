import argparse
import bisect
import os
import pickle
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    dataset: str             # "mnist" | "imagenet64"
    data_dir: str
    imagenet_dir: str
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    seed: int
    num_workers: int
    val_ratio: float
    log_interval: int
    realtime_progress: bool
    imagenet_cache_size: int
    save_path: str

    # TensorBoard
    tb_logdir: str
    no_tensorboard: bool

    # Augmentation
    use_rrc: bool  # RandomResizedCrop (ImageNet64 train)

    # Scheduler
    use_scheduler: bool
    step_size: int
    gamma: float


# ----------------------------
# ResNet-50
# ----------------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        layers: list[int],
        num_classes: int = 1000,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.inplanes = 64

        # ResNet stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stages
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        outplanes = planes * block.expansion

        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, outplanes, stride=stride),
                nn.BatchNorm2d(outplanes),
            )

        layers: list[nn.Module] = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50(in_channels: int, num_classes: int) -> ResNet:
    # ResNet-50 layers: [3, 4, 6, 3]
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.dataset == "mnist":
        return resnet50(in_channels=1, num_classes=10)
    return resnet50(in_channels=3, num_classes=1000)


# ----------------------------
# ImageNet64 Pickle Dataset
# ----------------------------
class ImageNet64PickleDataset(Dataset):
    def __init__(
        self,
        batch_files: Sequence[Path],
        transform: Optional[Callable] = None,
        cache_size: int = 1,
    ) -> None:
        self.batch_files = [Path(p) for p in batch_files]
        self.transform = transform
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self._lengths: list[int] = []
        for bf in self.batch_files:
            self._lengths.append(self._read_length(bf))
        self._cumulative = np.cumsum(self._lengths).tolist()

    @staticmethod
    def _read_length(batch_file: Path) -> int:
        with batch_file.open("rb") as f:
            payload = pickle.load(f)
        return len(payload["labels"])

    @staticmethod
    def _load_batch(batch_file: Path) -> tuple[np.ndarray, np.ndarray]:
        with batch_file.open("rb") as f:
            payload = pickle.load(f)
        data = np.asarray(payload["data"])
        labels = np.asarray(payload["labels"], dtype=np.int64) - 1  # 1~1000 -> 0~999
        if data.ndim != 2 or data.shape[1] != 64 * 64 * 3:
            raise ValueError(f"Unexpected batch shape in {batch_file}: {data.shape}")
        return data, labels

    @property
    def file_ranges(self) -> list[tuple[int, int]]:
        ranges = []
        start = 0
        for ln in self._lengths:
            end = start + ln
            ranges.append((start, end))
            start = end
        return ranges

    def _get_batch(self, file_idx: int) -> tuple[np.ndarray, np.ndarray]:
        bf = self.batch_files[file_idx]
        cached = self._cache.get(bf)
        if cached is not None:
            self._cache.move_to_end(bf)
            return cached
        data, labels = self._load_batch(bf)
        self._cache[bf] = (data, labels)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data, labels

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range (size={len(self)})")

        file_idx = bisect.bisect_right(self._cumulative, index)
        start = 0 if file_idx == 0 else self._cumulative[file_idx - 1]
        local_idx = index - start

        data, labels = self._get_batch(file_idx)
        img = data[local_idx].reshape(3, 64, 64).transpose(1, 2, 0)  # (H,W,C)
        pil = Image.fromarray(img.astype(np.uint8))
        y = int(labels[local_idx])

        if self.transform is not None:
            x = self.transform(pil)
        else:
            x = transforms.ToTensor()(pil)
        return x, y


class FileAwareShuffleSampler(Sampler[int]):
    def __init__(self, file_ranges: Sequence[tuple[int, int]], seed: int) -> None:
        self.file_ranges = list(file_ranges)
        self.seed = seed
        self.epoch = 0
        self._length = sum(e - s for s, e in self.file_ranges)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        file_order = torch.randperm(len(self.file_ranges), generator=g).tolist()
        for fi in file_order:
            s, e = self.file_ranges[fi]
            local_perm = torch.randperm(e - s, generator=g).tolist()
            for li in local_perm:
                yield s + li

    def __len__(self) -> int:
        return self._length


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_compact_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def print_realtime_progress(
    phase: str,
    step: int,
    total_steps: int,
    avg_loss: float,
    avg_acc: float,
    start_time: float,
) -> None:
    elapsed = time.perf_counter() - start_time
    it_per_sec = step / elapsed if elapsed > 0 else 0.0
    eta = (total_steps - step) / it_per_sec if it_per_sec > 0 else 0.0
    pct = int((step / max(total_steps, 1)) * 100)
    line = (
        f"\r[{phase}] loss: {avg_loss:.4f}, acc: {avg_acc:.4f} "
        f"{pct:3d}% {step}/{total_steps} "
        f"[{format_compact_duration(elapsed)}<{format_compact_duration(eta)}, {it_per_sec:5.2f}it/s]"
    )
    print(line, end="\n" if step == total_steps else "", flush=True)


# ----------------------------
# Data loaders
# ----------------------------
def _batch_sort_key(path: Path) -> int:
    suffix = path.name.rsplit("_", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def build_mnist_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=tf)

    val_size = int(len(ds) * cfg.val_ratio)
    train_size = len(ds) - val_size
    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(ds, [train_size, val_size], generator=gen)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    print(f"MNIST split | train: {len(train_set)} | val: {len(val_set)}")
    return train_loader, val_loader


def build_imagenet64_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    image_dir = Path(cfg.imagenet_dir)
    train_files = sorted(image_dir.glob("train_data_batch_*"), key=_batch_sort_key)
    val_file = image_dir / "val_data"

    if not train_files:
        raise FileNotFoundError(f"No train_data_batch_* found in {image_dir}")
    if not val_file.exists():
        raise FileNotFoundError(f"val_data not found in {image_dir}")

    # NOTE: 베스트는 네 데이터셋에서 mean/std 계산해서 넣기.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if cfg.use_rrc:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = ImageNet64PickleDataset(train_files, transform=train_tf, cache_size=cfg.imagenet_cache_size)
    val_ds = ImageNet64PickleDataset([val_file], transform=val_tf, cache_size=1)

    train_sampler = FileAwareShuffleSampler(train_ds.file_ranges, seed=cfg.seed)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    if cfg.dataset == "mnist":
        return build_mnist_loaders(cfg)
    return build_imagenet64_loaders(cfg)


# ----------------------------
# Train / Eval + TensorBoard
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    realtime_progress: bool,
    writer: Optional[SummaryWriter],
    global_step: int,
) -> Tuple[float, float, int]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    t0 = time.perf_counter()
    total_steps = len(loader)

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        running_correct += (out.argmax(dim=1) == y).sum().item()
        running_total += bs

        avg_loss = running_loss / max(running_total, 1)
        avg_acc = running_correct / max(running_total, 1)

        if writer is not None and log_interval > 0 and (batch_idx % log_interval == 0):
            writer.add_scalar("batch/train_loss", avg_loss, global_step)
            writer.add_scalar("batch/train_acc", avg_acc, global_step)
            writer.add_scalar("batch/lr", optimizer.param_groups[0]["lr"], global_step)

        if realtime_progress:
            print_realtime_progress("Training", batch_idx, total_steps, avg_loss, avg_acc, t0)

        global_step += 1

    return (running_loss / max(running_total, 1),
            running_correct / max(running_total, 1),
            global_step)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (out.argmax(dim=1) == y).sum().item()
        total_samples += bs

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


# ----------------------------
# Args / Main
# ----------------------------
def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train ResNet-50 with TensorBoard (MNIST or ImageNet64 pickle)")
    p.add_argument("--dataset", choices=["mnist", "imagenet", "imagenet64"], default="mnist")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--imagenet-dir", default="./Imagenet")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--no-realtime-progress", action="store_true")
    p.add_argument("--imagenet-cache-size", type=int, default=1)
    p.add_argument("--save-path", default="./resnet50_last.pth")

    p.add_argument("--tb-logdir", default="./runs/resnet50_exp")
    p.add_argument("--no-tensorboard", action="store_true")

    p.add_argument("--rrc", action="store_true", help="use RandomResizedCrop for ImageNet64 training")

    # Scheduler (기본 OFF 추천)
    p.add_argument("--use-scheduler", action="store_true")
    p.add_argument("--step-size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)

    args = p.parse_args()
    dataset = "imagenet64" if args.dataset == "imagenet" else args.dataset

    return TrainConfig(
        dataset=dataset,
        data_dir=args.data_dir,
        imagenet_dir=args.imagenet_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        log_interval=args.log_interval,
        realtime_progress=not args.no_realtime_progress,
        imagenet_cache_size=args.imagenet_cache_size,
        save_path=args.save_path,
        tb_logdir=args.tb_logdir,
        no_tensorboard=args.no_tensorboard,
        use_rrc=args.rrc,
        use_scheduler=args.use_scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    print(f"Model: ResNet-50 | Dataset: {cfg.dataset} | Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    scheduler = None
    if cfg.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    writer = None
    if not cfg.no_tensorboard:
        Path(cfg.tb_logdir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=cfg.tb_logdir)
        writer.add_text("hparams", str(cfg))
        # graph(무거우면 실패할 수 있어서 try)
        try:
            x0, _ = next(iter(train_loader))
            writer.add_graph(model, x0.to(device))
        except Exception:
            pass

    global_step = 0
    train_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=cfg.log_interval,
            realtime_progress=cfg.realtime_progress,
            writer=writer,
            global_step=global_step,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/train_acc", train_acc, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/val_acc", val_acc, epoch)
            writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)
            writer.flush()

        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - train_start
        overall_progress = epoch / max(cfg.epochs, 1)
        avg_epoch_time = total_elapsed / max(epoch, 1)
        total_eta = avg_epoch_time * (cfg.epochs - epoch)

        print(
            f"Epoch {epoch}/{cfg.epochs} ({overall_progress*100:5.1f}%) | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | "
            f"Epoch Time {format_duration(epoch_elapsed)} | "
            f"Total {format_duration(total_elapsed)} | "
            f"ETA {format_duration(total_eta)}"
        )

    if cfg.save_path:
        torch.save(model.state_dict(), cfg.save_path)
        print(f"Saved checkpoint to {cfg.save_path}")

    if writer is not None:
        writer.close()

    print("Finished training.")


if __name__ == "__main__":
    main()