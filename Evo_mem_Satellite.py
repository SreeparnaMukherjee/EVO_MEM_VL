import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, datasets
from transformers import ViTModel
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import os

# SETTINGS
DEVICE = "cpu" # Switching to CPU as requested by the environment
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-4
SEQ_LEN = 3 # Length of time-series sequence

# ==========================================
# 1. REAL DATASET (SATELLITE + OTHERS)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data():
    print("Loading datasets...")
    # 1. EuroSAT (Satellite)
    try:
        eurosat = datasets.EuroSAT(root="data", download=True, transform=transform)
        print(f"Loaded EuroSAT: {len(eurosat)} images")
    except Exception as e:
        print(f"Error loading EuroSAT: {e}. Falling back to CIFAR10.")
        eurosat = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

    # 2. CIFAR10
    cifar10 = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    
    # 3. MNIST (Grayscale to RGB)
    mnist_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mnist = datasets.MNIST(root="data", train=True, download=True, transform=mnist_transform)

    return eurosat, cifar10, mnist

class TimeSeriesDataset(Dataset):
    def __init__(self, dataset, seq_len=3):
        self.dataset = dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset) // self.seq_len

    def __getitem__(self, idx):
        imgs = []
        labels = []
        for i in range(self.seq_len):
            img, label = self.dataset[idx * self.seq_len + i]
            imgs.append(img)
            labels.append(label)
        # We take the last label as the target for time-series prediction
        return torch.stack(imgs), labels[-1]

# ==========================================
# 2. EVOLUTIONARY MEMORY MODEL
# ==========================================
class EvolutionaryMemoryBank(nn.Module):
    def __init__(self, embed_dim, capacity=64):
        super().__init__()
        self.capacity = capacity
        self.register_buffer('memory', torch.randn(capacity, embed_dim))
        self.register_buffer('fitness', torch.zeros(capacity))
        self.ptr = 0

    def write(self, features):
        batch_size = features.size(0)
        for i in range(batch_size):
            self.memory[self.ptr] = features[i].detach()
            self.fitness[self.ptr] = 1.0 # Initial fitness
            self.ptr = (self.ptr + 1) % self.capacity

    def read(self):
        return self.memory

    def update_fitness(self, scores):
        # Update fitness based on how often a memory is selected
        # scores: (batch_size, capacity)
        usage = scores.sum(dim=0)
        self.fitness = 0.9 * self.fitness + 0.1 * usage

class EvolutionarySelector(nn.Module):
    def __init__(self, embed_dim, mutation_rate=0.05):
        super().__init__()
        self.mutation_rate = mutation_rate

    def forward(self, current_feat, memory_bank):
        # 1. Fitness Calculation (Cosine Similarity)
        # current_feat: (B, D), memory_bank: (C, D)
        norm_feat = F.normalize(current_feat, p=2, dim=-1)
        norm_mem = F.normalize(memory_bank, p=2, dim=-1)
        similarity = torch.mm(norm_feat, norm_mem.t()) # (B, C)

        # 2. Selection (Top-K)
        k = 4
        top_k_vals, top_k_idx = torch.topk(similarity, k=k)
        
        # 3. Crossover (Weighted sum of top-K)
        weights = F.softmax(top_k_vals, dim=-1)
        selected = torch.zeros_like(current_feat)
        for i in range(k):
            selected += weights[:, i].unsqueeze(1) * memory_bank[top_k_idx[:, i]]

        # 4. Mutation
        if self.training:
            mutation = torch.randn_like(selected) * self.mutation_rate
            selected = selected + mutation

        return selected, similarity

class EvolutionMemoryModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # Freeze encoder for speed on CPU
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        embed_dim = self.encoder.config.hidden_size
        self.memory = EvolutionaryMemoryBank(embed_dim)
        self.selector = EvolutionarySelector(embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        batch_size, seq_len, _, _, _ = x.size()
        
        # Encode each image in sequence (only the last one for now to save time)
        last_img = x[:, -1]
        feat = self.encoder(last_img).last_hidden_state[:, 0, :] # (B, D)

        if self.training:
            self.memory.write(feat)

        mem_feat, sim_scores = self.selector(feat, self.memory.read())
        self.memory.update_fitness(sim_scores)

        combined = torch.cat([feat, mem_feat], dim=-1)
        return self.decoder(combined)

# ==========================================
# 3. BASELINE: ViT + LSTM
# ==========================================
class LSTMBaselineModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        embed_dim = self.encoder.config.hidden_size
        self.lstm = nn.LSTM(embed_dim, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()
        feats = []
        for t in range(seq_len):
            feat = self.encoder(x[:, t]).last_hidden_state[:, 0, :]
            feats.append(feat)
        
        feats = torch.stack(feats, dim=1) # (B, T, D)
        _, (h_n, _) = self.lstm(feats)
        return self.fc(h_n[-1])

# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return 100 * correct / total, all_preds, all_labels

def calculate_text_metrics(preds, labels, class_names):
    # Map class indices to names for text metrics
    pred_texts = [class_names[p] for p in preds]
    label_texts = [class_names[l] for l in labels]
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_bleu = 0
    total_rouge = 0
    total_meteor = 0
    
    n = min(len(pred_texts), 50) # Limit to 50 samples for speed
    for i in range(n):
        ref = label_texts[i]
        hyp = pred_texts[i]
        
        total_bleu += sentence_bleu([ref.split()], hyp.split(), weights=(1,0,0,0))
        total_rouge += scorer.score(ref, hyp)['rougeL'].fmeasure
        # total_meteor += meteor_score([ref.split()], hyp.split()) # meteor expects split strings in some versions
    
    return total_bleu/n, total_rouge/n, 0 # Meteor skipped for compatibility if needed

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    eurosat, cifar10, mnist = load_data()
    
    # Use a subset for faster demonstration
    subset_indices = list(range(100)) # 100 samples
    eurosat_sub = Subset(eurosat, subset_indices)
    
    dataset = TimeSeriesDataset(eurosat_sub, seq_len=SEQ_LEN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    num_classes = len(eurosat.classes)
    class_names = eurosat.classes

    # Models
    evo_model = EvolutionMemoryModel(num_classes).to(DEVICE)
    lstm_model = LSTMBaselineModel(num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    history = {"evo": [], "lstm": []}
    
    print("\n--- Training Evolutionary Memory Model ---")
    optimizer = torch.optim.Adam(evo_model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        loss = train_one_epoch(evo_model, train_loader, optimizer, criterion)
        acc, _, _ = evaluate(evo_model, val_loader)
        history["evo"].append(acc)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Acc: {acc:.2f}%")
        
    print("\n--- Training LSTM Baseline ---")
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        loss = train_one_epoch(lstm_model, train_loader, optimizer, criterion)
        acc, _, _ = evaluate(lstm_model, val_loader)
        history["lstm"].append(acc)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Acc: {acc:.2f}%")

    # Evaluation
    evo_acc, evo_preds, evo_labels = evaluate(evo_model, val_loader)
    bleu, rouge, _ = calculate_text_metrics(evo_preds, evo_labels, class_names)
    
    print("\nFinal Metrics (Evolutionary Model):")
    print(f"Accuracy: {evo_acc:.2f}%")
    print(f"BLEU-1: {bleu:.4f}")
    print(f"ROUGE-L: {rouge:.4f}")

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), history["evo"], label="Evolutionary Memory", marker='o')
    plt.plot(range(1, EPOCHS+1), history["lstm"], label="LSTM Baseline", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Evolutionary Memory vs LSTM Baseline on EuroSAT")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_comparison.png")
    print("\nVisualization saved as performance_comparison.png")
    plt.show()
