import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from guider.evaluation import seed_inference, write_json
from guider.models.pgm import ImprovedDualTaskTransformer


ROOT = Path(__file__).resolve().parent


def load_data(paths):
    arrays = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            arrays.append(data[data.files[0]])
    values = np.concatenate(arrays)
    if values.ndim != 2 or values.shape[1] != 40 or not np.isfinite(values).all():
        raise ValueError("dataset.shape / finite")
    labels = values[:, 39]
    if not ((labels == labels.astype(np.int64)) & (labels >= 0) & (labels < 9)).all():
        raise ValueError("dataset.labels")
    return values[:, :39].astype(np.float32), labels.astype(np.int64)


@torch.inference_mode()
def evaluate(model, loader, device, fine_threshold, coarse_threshold):
    model.eval()
    total = fine_correct = coarse_correct = fine_accepted = coarse_accepted = accepted_correct = 0
    for states, labels in loader:
        states, labels = states.to(device), labels.to(device)
        fine, coarse = model(states)
        fine_probs, coarse_probs = fine.softmax(1), coarse.softmax(1)
        fine_values, fine_actions = fine_probs.max(1)
        coarse_values, coarse_actions = coarse_probs.max(1)
        use_fine = fine_values >= fine_threshold
        use_coarse = (~use_fine) & (coarse_values >= coarse_threshold)
        fine_correct += int((fine_actions == labels).sum())
        coarse_correct += int((coarse_actions == labels % 3).sum())
        fine_accepted += int(use_fine.sum())
        coarse_accepted += int(use_coarse.sum())
        accepted_correct += int(((fine_actions == labels) & use_fine).sum())
        accepted_correct += int(((coarse_actions == labels % 3) & use_coarse).sum())
        total += len(labels)
    accepted = fine_accepted + coarse_accepted
    return {"samples": total, "fine_accuracy": fine_correct / total,
            "coarse_accuracy": coarse_correct / total, "fine_accepted": fine_accepted,
            "coarse_accepted": coarse_accepted, "rejected": total - accepted,
            "accepted_accuracy": accepted_correct / accepted if accepted else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", type=Path, default=[ROOT / "data/pgm.npz"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pgm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine-threshold", type=float, default=0.9)
    parser.add_argument("--coarse-threshold", type=float, default=0.9)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        parser.error("epochs / batch-size / threads / learning-rate")
    if not 0 <= args.fine_threshold <= 1 or not 0 <= args.coarse_threshold <= 1:
        parser.error("fine-threshold / coarse-threshold")
    torch.set_num_threads(args.threads)
    seed_inference(args.seed)
    states, labels = load_data(args.dataset)
    indices = np.arange(len(labels))
    train_indices, remaining = train_test_split(indices, test_size=0.4, random_state=args.seed, stratify=labels % 3)
    val_indices, test_indices = train_test_split(remaining, test_size=0.5, random_state=args.seed,
                                                stratify=labels[remaining] % 3)
    make_loader = lambda selected, shuffle: DataLoader(
        TensorDataset(torch.from_numpy(states[selected]), torch.from_numpy(labels[selected])),
        batch_size=args.batch_size, shuffle=shuffle)
    train_loader = make_loader(train_indices, True)
    val_loader = make_loader(val_indices, False)
    test_loader = make_loader(test_indices, False)
    model = ImprovedDualTaskTransformer().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", vars(args))
    np.savez_compressed(args.output / "split_indices.npz", train=train_indices, validation=val_indices, test=test_indices)
    best_accuracy = -1.0
    with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            fine_weight = 0.5 + 0.3 * epoch / args.epochs
            for states_batch, labels_batch in train_loader:
                states_batch, labels_batch = states_batch.to(args.device), labels_batch.to(args.device)
                states_batch = states_batch + torch.randn_like(states_batch) * 0.01
                fine, coarse = model(states_batch)
                loss = (fine_weight * F.cross_entropy(fine, labels_batch)
                        + (1 - fine_weight) * F.cross_entropy(coarse, labels_batch % 3))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * len(labels_batch)
            metrics = evaluate(model, val_loader, args.device, args.fine_threshold, args.coarse_threshold)
            metrics.update(epoch=epoch + 1, loss=total_loss / len(train_indices))
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(metrics), flush=True)
            if metrics["fine_accuracy"] > best_accuracy:
                best_accuracy = metrics["fine_accuracy"]
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                            "fine_threshold": args.fine_threshold, "coarse_threshold": args.coarse_threshold,
                            "validation_fine_accuracy": best_accuracy}, args.output / "best_model.pth")
    state = torch.load(args.output / "best_model.pth", map_location=args.device, weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    write_json(args.output / "test.json", evaluate(model, test_loader, args.device,
                                                  args.fine_threshold, args.coarse_threshold))


if __name__ == "__main__":
    main()
