# EvoMem-VL: Evolutionary Episodic Memory-Augmented Vision-Language Model for Satellite Image Captioning
A novel lifelong learning architecture that combines evolutionary algorithms with episodic memory and self-rewriting mechanisms to enable transformers to learn continuously across multiple tasks without catastrophic forgetting.

Architecture Overview

Input Sequence / Task
        │
        ▼
Transformer Encoder
        │
        ▼
Context Representation
        │
        ▼
Episodic Memory Interface
   ┌────┴─────┬──────────────┐
   ▼          ▼              ▼
Memory    Evolution     Self-Rewriting
Retrieval  Engine         Module
   └────┬─────┴──────────────┘
        │
        ▼
Memory-Enhanced Transformer
        │
        ▼
  Output Generator

Datasets

Three benchmark datasets are used, spanning both vision and NLP domains:

1. Split-MNIST

Type: Image classification
Size: 70,000 grayscale images (28×28)
Classes: 10 digit classes (0–9) split into 5 tasks × 2 classes
Input dim: 784 (flattened)
Source: torchvision.datasets.MNIST (auto-download)

2. Split-CIFAR-10

Type: Image classification
Size: 60,000 RGB images (32×32×3)
Classes: 10 object classes split into 5 tasks × 2 classes
Input dim: 3,072 (flattened)
Source: torchvision.datasets.CIFAR10 (auto-download)

3. CLINC150

Type: Natural language intent classification
Size: 23,700 utterances across 150 intent classes
Classes: 150 intents split into 5 tasks × 30 classes
Input dim: 300 (TF-IDF bag-of-words embedding)
Source: clinc/oos-eval (auto-download with synthetic fallback)
