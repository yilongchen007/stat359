import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        self.centers = torch.tensor(skipgram_df["center"].values, dtype=torch.long)
        self.contexts = torch.tensor(skipgram_df["context"].values, dtype=torch.long)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, context_words):
        center_vectors = self.in_embed(center_words)
        context_vectors = self.out_embed(context_words)
        return (center_vectors * context_vectors).sum(dim=1)

    def get_embeddings(self):
        return self.in_embed.weight.detach().cpu().numpy()


# Load processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

skipgram_df = data["skipgram_df"]
word2idx = data["word2idx"]
counter = data["counter"]
vocab_size = len(word2idx)

# Precompute negative sampling distribution below
word_counts = torch.zeros(vocab_size, dtype=torch.float)
for word, idx in word2idx.items():
    word_counts[idx] = counter.get(word, 0)

unigram_dist = word_counts / word_counts.sum()
negative_sampling_dist = unigram_dist.pow(0.75)
negative_sampling_dist = negative_sampling_dist / negative_sampling_dist.sum()


def sample_negative(contexts, distribution, num_samples):
    batch_size = contexts.size(0)
    neg_samples = torch.multinomial(
        distribution, batch_size * num_samples, replacement=True
    ).view(batch_size, num_samples)
    mask = neg_samples.eq(contexts.unsqueeze(1))
    while mask.any():
        neg_samples[mask] = torch.multinomial(
            distribution, int(mask.sum()), replacement=True
        )
        mask = neg_samples.eq(contexts.unsqueeze(1))
    return neg_samples


# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def make_targets(center, context, vocab_size):
    targets = torch.zeros(center.size(0), vocab_size, device=center.device)
    targets[torch.arange(center.size(0)), context] = 1.0
    return targets


# Training loop
negative_sampling_dist = negative_sampling_dist.to(device)
for epoch in range(EPOCHS):
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for centers, contexts in progress:
        centers = centers.to(device)
        contexts = contexts.to(device)

        optimizer.zero_grad()

        pos_logits = model(centers, contexts)
        pos_labels = torch.ones_like(pos_logits)
        pos_loss = criterion(pos_logits, pos_labels)

        negative_contexts = sample_negative(
            contexts, negative_sampling_dist, NEGATIVE_SAMPLES
        ).to(device)
        center_vectors = model.in_embed(centers)
        negative_vectors = model.out_embed(negative_contexts)
        neg_logits = torch.bmm(
            negative_vectors, center_vectors.unsqueeze(2)
        ).squeeze(2)
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = criterion(neg_logits, neg_labels)

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")


# Save embeddings and mappings
embeddings = model.get_embeddings()
with open("word2vec_embeddings.pkl", "wb") as f:
    pickle.dump(
        {"embeddings": embeddings, "word2idx": data["word2idx"], "idx2word": data["idx2word"]},
        f,
    )
print("Embeddings saved to word2vec_embeddings.pkl")