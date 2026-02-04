import argparse
import bisect
import os
import pickle
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from torchvision import datasets, transforms


@dataclass
class TrainConfig:
    dataset: str
    data_dir: str
    imagenet_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    num_workers: int
    val_ratio: float
    log_interval: int
    realtime_progress: bool
    imagenet_cache_size: int
    save_path: str


class AlexNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ImageNet64PickleDataset(Dataset):
    def __init__(
        self,
        batch_files: Sequence[Path],
        transform: Optional[Callable] = None,
        cache_size: int = 1,
    ) -> None:
        self.batch_files = [Path(path) for path in batch_files]
        self.transform = transform
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self._lengths: list[int] = []
        for batch_file in self.batch_files:
            self._lengths.append(self._read_length(batch_file))
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

        data = payload["data"]
        labels = np.asarray(payload["labels"], dtype=np.int64) - 1
        if data.ndim != 2 or data.shape[1] != 64 * 64 * 3:
            raise ValueError(f"Unexpected ImageNet64 batch shape in {batch_file}: {data.shape}")
        return data, labels

    @property
    def file_ranges(self) -> list[tuple[int, int]]:
        ranges = []
        start = 0
        for length in self._lengths:
            end = start + length
            ranges.append((start, end))
            start = end
        return ranges

    def _get_batch(self, file_idx: int) -> tuple[np.ndarray, np.ndarray]:
        batch_file = self.batch_files[file_idx]
        cached = self._cache.get(batch_file)
        if cached is not None:
            self._cache.move_to_end(batch_file)
            return cached

        data, labels = self._load_batch(batch_file)
        self._cache[batch_file] = (data, labels)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data, labels

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with size {len(self)}")

        file_idx = bisect.bisect_right(self._cumulative, index)
        start = 0 if file_idx == 0 else self._cumulative[file_idx - 1]
        local_idx = index - start

        data, labels = self._get_batch(file_idx)
        image = data[local_idx].reshape(3, 64, 64).transpose(1, 2, 0)
        image = Image.fromarray(image, mode="RGB")
        label = int(labels[local_idx])

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FileAwareShuffleSampler(Sampler[int]):
    def __init__(self, file_ranges: Sequence[tuple[int, int]], seed: int) -> None:
        self.file_ranges = list(file_ranges)
        self.seed = seed
        self.epoch = 0
        self._length = sum(end - start for start, end in self.file_ranges)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        file_order = torch.randperm(len(self.file_ranges), generator=generator).tolist()

        for file_idx in file_order:
            start, end = self.file_ranges[file_idx]
            local_perm = torch.randperm(end - start, generator=generator).tolist()
            for local_idx in local_perm:
                yield start + local_idx

    def __len__(self) -> int:
        return self._length


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
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def format_compact_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


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
        f"\r[{phase}] loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}: "
        f"{pct:3d}% {step}/{total_steps} "
        f"[{format_compact_duration(elapsed)}<{format_compact_duration(eta)}, {it_per_sec:5.2f}it/s]"
    )
    print(line, end="\n" if step == total_steps else "", flush=True)


def save_accuracy_plot(history: dict[str, list[float]], plot_path: str) -> None:
    if not history["train_acc"]:
        return

    os.environ.setdefault("MPLCONFIGDIR", str((Path(".") / ".matplotlib").resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not save plot: {exc}")
        return

    epochs = list(range(1, len(history["train_acc"]) + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_acc"], marker="o", linewidth=2, label="Train Acc")
    ax.plot(epochs, history["val_acc"], marker="o", linewidth=2, label="Val Acc")
    ax.set_title("Train vs Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Saved accuracy graph to {plot_file}")


def save_loss_plot(history: dict[str, list[float]], plot_path: str) -> None:
    if not history["train_loss"]:
        return

    os.environ.setdefault("MPLCONFIGDIR", str((Path(".") / ".matplotlib").resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not save plot: {exc}")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"], marker="o", linewidth=2, label="Val Loss")
    ax.set_title("Train vs Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Saved loss graph to {plot_file}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    realtime_progress: bool,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    epoch_start = time.perf_counter()
    total_steps = len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == targets).sum().item()
        running_total += batch_size

        avg_loss = running_loss / max(running_total, 1)
        avg_acc = running_correct / max(running_total, 1)
        if realtime_progress:
            print_realtime_progress("Training", batch_idx, total_steps, avg_loss, avg_acc, epoch_start)
        else:
            should_log = (
                log_interval > 0
                and (batch_idx % log_interval == 0 or batch_idx == total_steps)
            )
            if should_log:
                progress = batch_idx / max(total_steps, 1)
                elapsed = time.perf_counter() - epoch_start
                eta = elapsed / max(progress, 1e-8) - elapsed
                print(
                    f"Epoch {epoch} | Step {batch_idx}/{total_steps} ({progress * 100:5.1f}%) | "
                    f"Loss {avg_loss:.4f} | Acc {avg_acc:.4f} | "
                    f"Elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
                )

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def build_mnist_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
    val_size = int(len(dataset) * cfg.val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    print(f"MNIST split | train: {len(train_set)} | val: {len(val_set)}")
    return train_loader, val_loader


def _batch_sort_key(path: Path) -> int:
    suffix = path.name.rsplit("_", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def build_imagenet64_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    image_dir = Path(cfg.imagenet_dir)
    train_files = sorted(image_dir.glob("train_data_batch_*"), key=_batch_sort_key)
    val_file = image_dir / "val_data"
    if not val_file.exists():
        raise FileNotFoundError(f"val_data not found in {image_dir}")

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    train_dataset = ImageNet64PickleDataset(train_files, transform=train_transform, cache_size=cfg.imagenet_cache_size)
    val_dataset   = ImageNet64PickleDataset([val_file], transform=val_transform, cache_size=1)

    train_sampler = FileAwareShuffleSampler(train_dataset.file_ranges, seed=cfg.seed)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_loader   = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    if cfg.dataset == "mnist":
        return build_mnist_loaders(cfg)
    return build_imagenet64_loaders(cfg)


def build_model(cfg: TrainConfig) -> AlexNet:
    if cfg.dataset == "mnist":
        return AlexNet(in_channels=1, num_classes=10)
    return AlexNet(in_channels=3, num_classes=1000)


def parse_args() -> tuple[TrainConfig, str, str, bool]:
    parser = argparse.ArgumentParser(description="Train AlexNet on MNIST or ImageNet64")
    parser.add_argument("--dataset", choices=["mnist", "imagenet", "imagenet64"], default="mnist")
    parser.add_argument("--data-dir", default="./data", help="MNIST root directory")
    parser.add_argument("--imagenet-dir", default="./Imagenet", help="ImageNet64 batch directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--no-realtime-progress",
        action="store_true",
        help="disable one-line real-time batch progress output",
    )
    parser.add_argument(
        "--imagenet-cache-size",
        type=int,
        default=1,
        help="Number of ImageNet64 batch files to keep in RAM cache",
    )
    parser.add_argument("--save-path", default="./alexnet_last.pth", help="model checkpoint path")
    parser.add_argument("--plot-path", default="./alexnet_acc_compare.png", help="train-vs-val accuracy graph path")
    parser.add_argument("--loss-plot-path", default="./alexnet_loss_compare.png", help="train-vs-val loss graph path")
    parser.add_argument("--no-plot", action="store_true", help="disable saving graphs")
    args = parser.parse_args()

    dataset = "imagenet64" if args.dataset == "imagenet" else args.dataset
    cfg = TrainConfig(
        dataset=dataset,
        data_dir=args.data_dir,
        imagenet_dir=args.imagenet_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        log_interval=args.log_interval,
        realtime_progress=not args.no_realtime_progress,
        imagenet_cache_size=args.imagenet_cache_size,
        save_path=args.save_path,
    )
    return cfg, args.plot_path, args.loss_plot_path, args.no_plot


def main() -> None:
    cfg, plot_path, loss_plot_path, no_plot = parse_args()
    set_seed(cfg.seed)

    os.makedirs(cfg.data_dir, exist_ok=True)
    device = get_device()
    print(f"Dataset: {cfg.dataset} | Device: {device}")

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    train_start = time.perf_counter()
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=cfg.log_interval,
            realtime_progress=cfg.realtime_progress,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - train_start
        overall_progress = epoch / max(cfg.epochs, 1)
        avg_epoch_time = total_elapsed / max(epoch, 1)
        total_eta = avg_epoch_time * (cfg.epochs - epoch)
        print(
            f"Epoch {epoch}/{cfg.epochs} complete ({overall_progress * 100:5.1f}%) | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | "
            f"Epoch Time {format_duration(epoch_elapsed)} | "
            f"Total Elapsed {format_duration(total_elapsed)} | "
            f"Total ETA {format_duration(total_eta)}"
        )

    if cfg.save_path:
        torch.save(model.state_dict(), cfg.save_path)
        print(f"Saved checkpoint to {cfg.save_path}")
    if not no_plot:
        save_accuracy_plot(history, plot_path)
        save_loss_plot(history, loss_plot_path)

    print("Finished training.")


if __name__ == "__main__":
    main()
