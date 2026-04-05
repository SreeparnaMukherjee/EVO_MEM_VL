# Evolutionary Memory Model for Satellite Image Captioning
**Overview:

EVO_MEM_VL (Evolution Memory Vision Learning) is a deep learning project that integrates Vision Transformers (ViT) with a novel evolutionary episodic memory mechanism to improve classification of temporal satellite imagery.

The model simulates real-world remote sensing datasets and introduces a memory-augmented learning paradigm, where past learned representations evolve and influence future predictions.
Architecture:

<img width="273" height="256" alt="image" src="https://github.com/user-attachments/assets/29d81eb5-eeb4-4863-9735-079c8f8b47bf" />



Key Features:

-Episodic Memory Bank (EMB)
Stores learned feature embeddings dynamically during training.

-Evolutionary Memory Selection
Uses similarity-based retrieval + mutation to evolve useful past knowledge.

-Vision Transformer Encoder (ViT)
Extracts high-level image representations using pretrained transformer architecture.

-Temporal Satellite Data Simulation
Mimics real-world datasets like:
NWPU (categorical classification)
RSICD (caption-based understanding)
Sentinel-2 (temporal evolution)

-Memory-Augmented Classification
Combines current features with retrieved memory embeddings for better predictions.


Component Breakdown:

Vision Transformer Encoder (ViT)
Splits each satellite image into 16×16 patches and encodes them into dense feature embeddings using a pretrained ViT-Base/16 backbone. The last two transformer blocks are fine-tuned while the rest of the backbone is frozen to preserve pretrained spatial representations.
Episodic Memory Bank (EMB).
A learnable key-value memory store with three distinct operations:

Memory Write — absorbs new visual features into memory slots via attention-gated writes, controlling how much new information each slot should store
Memory Read — retrieves relevant context from memory conditioned on the current image features through cross-attention
Memory Rewrite — applies a gated LSTM-style update to refresh stale memory slots using aggregated read context, preventing memory degradation over time

Evolutionary Selector
Applies genetic algorithm operations over the memory population after each forward pass:

Fitness Scoring — a learned network scores each memory slot by its relevance to the current image context using the [CLS] token representation
Selection — retains the top-K elite memory slots by fitness score
Crossover — pairs adjacent elite slots and blends them using a learned mixing coefficient α, generating offspring that inherit properties of two high-fitness memories
Mutation — adds structured learned noise during training to maintain population diversity and prevent premature convergence

Memory-Augmented Decoder
A Transformer decoder with dual cross-attention streams:

Standard cross-attention over ViT visual patch features
Additional cross-attention over the evolved memory population
A gated fusion layer that dynamically weighs how much each stream contributes to each generated token

Adaptive Semantic Beam Search
An enhanced beam search decoder with two diversity mechanisms:

Length penalty to prevent short degenerate outputs
Diversity penalty that discourages beams from repeating high-frequency tokens, producing more varied and semantically rich captions


Output:

-Training vs Validation Loss Graph

-Final prediction visualization of temporal image stream

-Caption-style prediction for detected theme
