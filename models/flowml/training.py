from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def regression_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().cpu().flatten()
    y = y.detach().cpu().flatten()
    mae = torch.mean(torch.abs(pred - y)).item()
    rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item()
    return {"mae": mae, "rmse": rmse}


def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1).detach().cpu()
    y = y.detach().cpu()
    accuracy = (pred == y).float().mean().item()
    classes = sorted(y.unique().tolist())
    f1s = []
    for cls in classes:
        tp = ((pred == cls) & (y == cls)).sum().item()
        fp = ((pred == cls) & (y != cls)).sum().item()
        fn = ((pred != cls) & (y == cls)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-8))
    return {"accuracy": accuracy, "macro_f1": float(sum(f1s) / max(len(f1s), 1))}


def train_supervised(
    model: nn.Module,
    train: Dataset,
    val: Dataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    task: str = "regression",
    graph_adjacency: torch.Tensor | None = None,
) -> dict[str, Any]:
    run_device = device()
    model = model.to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion: nn.Module = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    history = []
    best_state = None
    best_val_loss = float("inf")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)
    adjacency = graph_adjacency.to(run_device) if graph_adjacency is not None else None

    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(run_device).float()
            y = y.to(run_device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, adjacency) if adjacency is not None else model(x)
            loss = criterion(pred, y.long() if task == "classification" else y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)

        model.eval()
        val_loss = 0.0
        preds = []
        ys = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(run_device).float()
                y = y.to(run_device)
                pred = model(x, adjacency) if adjacency is not None else model(x)
                loss = criterion(pred, y.long() if task == "classification" else y.float())
                val_loss += loss.item() * len(x)
                preds.append(pred.detach().cpu())
                ys.append(y.detach().cpu())

        pred_all = torch.cat(preds) if preds else torch.empty(0)
        y_all = torch.cat(ys) if ys else torch.empty(0)
        metrics = (
            classification_metrics(pred_all, y_all)
            if task == "classification"
            else regression_metrics(pred_all, y_all)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(len(train), 1),
                "val_loss": val_loss / max(len(val), 1),
                "seconds": time.time() - started,
                **metrics,
            }
        )
        if history[-1]["val_loss"] < best_val_loss:
            best_val_loss = history[-1]["val_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "final": history[-1] if history else {},
        "best": min(history, key=lambda item: item["val_loss"]) if history else {},
    }


def train_masked_imputer(
    model: nn.Module,
    train: Dataset,
    val: Dataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    run_device = device()
    model = model.to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    best_state = None
    best_val_loss = float("inf")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        train_loss = 0.0
        for masked, target, observed_mask in train_loader:
            masked = masked.to(run_device)
            target = target.to(run_device)
            observed_mask = observed_mask.to(run_device)
            missing_mask = 1.0 - observed_mask
            optimizer.zero_grad(set_to_none=True)
            pred = model(masked, observed_mask)
            loss = (((pred - target) ** 2) * missing_mask).sum() / missing_mask.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(masked)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for masked, target, observed_mask in val_loader:
                masked = masked.to(run_device)
                target = target.to(run_device)
                observed_mask = observed_mask.to(run_device)
                missing_mask = 1.0 - observed_mask
                pred = model(masked, observed_mask)
                loss = (((pred - target) ** 2) * missing_mask).sum() / missing_mask.sum().clamp_min(1.0)
                val_loss += loss.item() * len(masked)
        item = {
            "epoch": epoch,
            "train_loss": train_loss / max(len(train), 1),
            "val_loss": val_loss / max(len(val), 1),
            "seconds": time.time() - started,
        }
        history.append(item)
        if item["val_loss"] < best_val_loss:
            best_val_loss = item["val_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "final": history[-1] if history else {}, "best": min(history, key=lambda item: item["val_loss"]) if history else {}}


def evaluate_masked_imputer(model: nn.Module, dataset: Dataset, *, batch_size: int) -> dict[str, float]:
    run_device = device()
    model = model.to(run_device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    with torch.no_grad():
        for masked, target, observed_mask in loader:
            masked = masked.to(run_device)
            target = target.to(run_device)
            observed_mask = observed_mask.to(run_device)
            missing_mask = 1.0 - observed_mask
            pred = model(masked, observed_mask)
            loss = (((pred - target) ** 2) * missing_mask).sum() / missing_mask.sum().clamp_min(1.0)
            total_loss += loss.item() * len(masked)
    return {"test_masked_mse": total_loss / max(len(dataset), 1)}


def evaluate_supervised(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    task: str = "regression",
    graph_adjacency: torch.Tensor | None = None,
) -> dict[str, float]:
    run_device = device()
    model = model.to(run_device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion: nn.Module = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    adjacency = graph_adjacency.to(run_device) if graph_adjacency is not None else None
    preds = []
    ys = []
    loss_total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(run_device).float()
            y = y.to(run_device)
            pred = model(x, adjacency) if adjacency is not None else model(x)
            loss = criterion(pred, y.long() if task == "classification" else y.float())
            loss_total += loss.item() * len(x)
            preds.append(pred.detach().cpu())
            ys.append(y.detach().cpu())
    pred_all = torch.cat(preds) if preds else torch.empty(0)
    y_all = torch.cat(ys) if ys else torch.empty(0)
    metrics = (
        classification_metrics(pred_all, y_all)
        if task == "classification"
        else regression_metrics(pred_all, y_all)
    )
    return {"test_loss": loss_total / max(len(dataset), 1), **{f"test_{k}": v for k, v in metrics.items()}}


def train_autoencoder(
    model: nn.Module,
    train: Dataset,
    val: Dataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    return train_supervised(
        model,
        train,
        val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        task="regression",
    )


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    save_file(model.state_dict(), path.with_suffix(".safetensors"))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def benchmark_latency(
    model: nn.Module,
    sample_args: tuple[torch.Tensor, ...],
    *,
    repeats: int = 50,
) -> dict[str, float]:
    run_device = device()
    model = model.to(run_device)
    model.eval()
    args = tuple(arg.to(run_device).float() for arg in sample_args)
    with torch.no_grad():
        for _ in range(5):
            _ = model(*args)
        if run_device.type == "cuda":
            torch.cuda.synchronize()
        started = time.time()
        for _ in range(repeats):
            _ = model(*args)
        if run_device.type == "cuda":
            torch.cuda.synchronize()
    return {
        "device": str(run_device),
        "batchSize": int(args[0].shape[0]),
        "meanLatencyMs": ((time.time() - started) / repeats) * 1000,
    }


def export_torchscript(path: Path, model: nn.Module, sample_args: tuple[torch.Tensor, ...]) -> bool:
    try:
        model_cpu = model.cpu().eval()
        traced = torch.jit.trace(
            model_cpu,
            tuple(arg.cpu().float() for arg in sample_args),
            strict=False,
            check_trace=False,
        )
        traced.save(str(path))
        return True
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")
        return False
