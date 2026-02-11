#!/usr/bin/env python
"""Train an MLP sentiment classifier with mean-pooled FastText embeddings."""

import argparse
import os
import random
import re
from dataclasses import dataclass

import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


DEFAULT_FASTTEXT_PATH = '../Assignment_2/fasttext-wiki-news-subwords-300.model'
CLASS_NAMES = ['Negative (0)', 'Neutral (1)', 'Positive (2)']


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select available accelerator."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def tokenize(text: str) -> list[str]:
    """A lightweight tokenizer for fastText lookup."""
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def load_fasttext(path: str) -> KeyedVectors:
    """Load local fastText vectors saved in Assignment 2."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'FastText model not found at {path}. '
            'Expected Assignment 2 artifact fasttext-wiki-news-subwords-300.model.'
        )
    print(f'Loading fastText vectors from: {path}')
    return KeyedVectors.load(path)


def sentence_mean_embedding(sentence: str, vectors: KeyedVectors, dim: int) -> np.ndarray:
    """Average token vectors into one sentence embedding."""
    tokens = tokenize(sentence)
    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    token_vectors = []
    for token in tokens:
        try:
            token_vectors.append(vectors.get_vector(token))
        except KeyError:
            continue

    if not token_vectors:
        return np.zeros(dim, dtype=np.float32)

    return np.mean(token_vectors, axis=0, dtype=np.float32)


def build_features(sentences: list[str], vectors: KeyedVectors, dim: int) -> np.ndarray:
    """Build matrix of mean-pooled sentence embeddings."""
    features = [sentence_mean_embedding(sent, vectors, dim) for sent in sentences]
    return np.asarray(features, dtype=np.float32)


def load_financial_phrasebank() -> tuple[np.ndarray, np.ndarray]:
    """Load sentence texts and labels."""
    ds = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
    sentences = np.asarray(ds['train']['sentence'])
    labels = np.asarray(ds['train']['label'], dtype=np.int64)
    return sentences, labels


def stratified_splits(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create 70/15/15 train/val/test splits with stratification."""
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=seed,
        stratify=y,
    )
    val_ratio = 0.15 / 0.85
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_trainval,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


class EmbeddingDataset(Dataset):
    """Torch dataset for dense embedding inputs."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """MLP for sentence-level sentiment classification."""

    def __init__(self, input_dim: int, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    f1_macro: float


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[EpochMetrics, np.ndarray, np.ndarray]:
    """Evaluate model on a dataloader."""
    model.eval()
    total_loss = 0.0
    preds_all: list[int] = []
    labels_all: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch_y.cpu().numpy())

    labels_arr = np.asarray(labels_all)
    preds_arr = np.asarray(preds_all)
    avg_loss = total_loss / len(loader.dataset)
    acc = float((preds_arr == labels_arr).mean())
    f1_macro = float(f1_score(labels_arr, preds_arr, average='macro'))
    return EpochMetrics(avg_loss, acc, f1_macro), labels_arr, preds_arr


def plot_curves(history: dict[str, list[float]], output_dir: str) -> None:
    """Save training/validation curves to disk."""
    epochs = np.arange(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 12))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP Loss Curve')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('MLP Accuracy Curve')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, history['train_f1'], label='Train Macro F1')
    plt.plot(epochs, history['val_f1'], label='Val Macro F1')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.title('MLP Macro F1 Curve')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mlp_learning_curves.png'), dpi=150)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str) -> None:
    """Save confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('MLP Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mlp_confusion_matrix.png'), dpi=150)
    plt.close()


def train(args: argparse.Namespace) -> None:
    """Full training + evaluation pipeline."""
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f'Using device: {device}')

    sentences, labels = load_financial_phrasebank()
    vectors = load_fasttext(args.fasttext_path)
    emb_dim = vectors.vector_size

    print('Building mean-pooled FastText sentence embeddings...')
    features = build_features(sentences.tolist(), vectors, emb_dim)

    x_train, y_train, x_val, y_val, x_test, y_test = stratified_splits(features, labels, args.seed)
    print(f'Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}')

    train_loader = DataLoader(EmbeddingDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EmbeddingDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = MLPClassifier(input_dim=emb_dim, dropout=args.dropout).to(device)

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
    )

    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    best_model_path = os.path.join(args.output_dir, 'best_mlp_model.pth')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds: list[int] = []
        train_labels: list[int] = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = float((np.asarray(train_preds) == np.asarray(train_labels)).mean())
        train_f1 = float(f1_score(train_labels, train_preds, average='macro'))

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics.f1_macro)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_metrics.loss)
        history['val_acc'].append(val_metrics.acc)
        history['val_f1'].append(val_metrics.f1_macro)

        print(
            f'Epoch {epoch:02d}/{args.epochs} | '
            f'Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | '
            f'Val Loss {val_metrics.loss:.4f} Acc {val_metrics.acc:.4f} F1 {val_metrics.f1_macro:.4f}'
        )

        if val_metrics.f1_macro > best_val_f1:
            best_val_f1 = val_metrics.f1_macro
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            stale_epochs += 1

        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(f'Early stopping at epoch {epoch}. Best epoch: {best_epoch}, best val F1: {best_val_f1:.4f}')
            break

    plot_curves(history, args.output_dir)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    print('\n========== Final Test Metrics (MLP) ==========')
    print(f'Accuracy   : {test_metrics.acc:.4f}')
    print(f'Macro F1   : {test_metrics.f1_macro:.4f}')
    print(f'Loss       : {test_metrics.loss:.4f}')
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    plot_confusion(y_true, y_pred, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP sentiment classifier (FastText mean pooling).')
    parser.add_argument('--fasttext-path', type=str, default=DEFAULT_FASTTEXT_PATH)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--min-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    train(parser.parse_args())
