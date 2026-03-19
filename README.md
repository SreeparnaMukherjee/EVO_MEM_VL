# Evolutionary Memory Model for Satellite Image Captioning
Overview:

This project implements a novel deep learning architecture for automatic satellite image captioning that combines episodic memory with evolutionary computation. The model ingests time-based streams of remote sensing imagery and generates natural language descriptions by learning what to remember, what to forget, and how to evolve its memory population over time.

Architecture:

<img width="273" height="256" alt="image" src="https://github.com/user-attachments/assets/29d81eb5-eeb4-4863-9735-079c8f8b47bf" />


Component Breakdown:
Vision Transformer Encoder (ViT)
Splits each satellite image into 16×16 patches and encodes them into dense feature embeddings using a pretrained ViT-Base/16 backbone. The last two transformer blocks are fine-tuned while the rest of the backbone is frozen to preserve pretrained spatial representations.
Episodic Memory Bank (EMB)
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


Datasets

<img width="547" height="163" alt="image" src="https://github.com/user-attachments/assets/274dfdfc-5db8-411c-bf95-04d53fcc0363" />

Results

The model is evaluated using standard image captioning metrics:

<img width="392" height="113" alt="image" src="https://github.com/user-attachments/assets/9ad995c3-c776-4f27-9eba-11e512733d23" />
