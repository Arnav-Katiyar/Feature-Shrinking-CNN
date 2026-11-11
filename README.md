# Feature-Shrinking CNN

A lightweight CNN built with TensorFlow/Keras that reduces the spatial dimensions of 28 × 28 grayscale images **without using any pooling layer**. Downsampling is achieved through a strided convolution and a `valid`-padded convolution.

## Architecture

Built with the Keras **Sequential API**:

| Layer | Configuration | Output Shape | Params |
|-------|---------------|--------------|--------|
| Input | grayscale image | (28, 28, 1) | 0 |
| Conv2D | 16 filters, 3×3, stride 2, `same`, ReLU | (14, 14, 16) | 160 |
| Conv2D | 32 filters, 3×3, stride 1, `valid`, ReLU | (12, 12, 32) | 4,640 |
| Flatten | — | (4608) | 0 |
| Dense | 10 units, softmax | (10) | 46,090 |

**Total parameters:** 50,890

### How the spatial size shrinks
- **Conv 1** (stride 2, `same`): 28 → ceil(28 / 2) = **14**
- **Conv 2** (stride 1, `valid`): 14 − 3 + 1 = **12**
- **Flatten:** 12 × 12 × 32 = **4608**

## Training Setup

- **Dataset:** MNIST (60,000 train / 10,000 test), pixels normalized to [0, 1]
- **Optimizer:** Adam
- **Loss:** Sparse categorical cross-entropy
- **Epochs:** 5, batch size 128, 10% validation split

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | ~98.49% |
| Test loss | ~0.0457 |

## Architecture Verification

The notebook includes assertions that check:
- Output shape after every layer
- No `MaxPooling2D` or `AveragePooling2D` layers are present

## Getting Started

### Requirements
- Python 3.9+
- TensorFlow
- NumPy
- Matplotlib

```bash
pip install tensorflow numpy matplotlib
```

### Run
```bash
jupyter notebook Feature_Shrinking_CNN.ipynb
```

Run all cells in order. The trained model is saved as `feature_shrinking_cnn.keras`.

## Files

| File | Description |
|------|-------------|
| `Feature_Shrinking_CNN.ipynb` | Model definition, training, evaluation and verification |
| `README.md` | Project overview |
