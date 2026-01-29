# Sparse Autoencoder for CLIP
This report outlines the implementation and evaluation of a **Sparse Autoencoder (SAE)** applied to CLIP image embeddings, with comparative experiments on Kandinsky-generated images.

## Training metrics

|Model| MSE | L1 | Explained variance | Dict size | Dict size / hidden size |
|-|-|-|-|-|-|
| SAE for CLIP | 0.0076 | 0.0031 | 0.9658 | 1533.8762 | 1.56% |
| SAE for Kandinsky | 0.0746 | 0.0081 | 0.9129 | 946.7031 | 0.58% |

## Classification metrics 

### Cifar10 dataset

| Model | Accuracy | Precision | Recall | F1 |
|-|-|-|-|-|
| CLIP | 95.35% | 95.47% | 95.35% | 95.36% |
| CLIP + SAE | 95.39% | 95.5% | 95.38% | 95.39% | 

### Cifar100 dataset

| Model | Accuracy | Precision | Recall | F1 |
|-|-|-|-|-|
| CLIP | 72.46% | 78.87% | 72.45% | 73.22% |
| CLIP + SAE | 72.49% | 78.28% | 72.48% | 73.18% |


## SAE Applied to Kandinsky Image Generation

### Experiment 1 – Original Image
![](/imgs/1.png)  
*Baseline image without modifications.*

**Identified features:**
- `131198` → South Italian place
- `103372` → Sweet blur in the distance
- `10667` → Street with a person
- `136354` → Light smooth texture


### Modified Results

| Image | Description | Modified Indices | Scaling Factors |
|-------|-------------|------------------|-----------------|
| ![](/imgs/2.png) | Enhanced Italian style | 131198 | 4× |
| ![](/imgs/3.png) | Added human figure and improved style | 103372, 10667, 136354 | 2×, 8×, 2× |


### Experiment 2 – Another Example
![](/imgs/4.png)  
*Second baseline image.*

**Identified features:**
- `143869` → Realistic style
- `2556` → Buildings in background
- `21083` → Close-up view
- `159602` → Minimalism


### Modified Results (Continued)

| Image | Description | Modified Indices | Scaling Factors |
|-------|-------------|------------------|-----------------|
| ![](/imgs/5.png) | Added human element from first experiment | 10667 | 12× |
| ![](/imgs/6.png) | Enhanced minimalism | 159602 | 4× |
| ![](/imgs/7.png) | Added buildings and improved style | 143869, 2556 | 4×, 8× |
| ![](/imgs/8.png) | Combined building, style, and close-up modifications | 143869, 2556, 21083 | 4×, 8×, 4× |
