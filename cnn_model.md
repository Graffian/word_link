# Word Link CNN — OCR Model Training Documentation

This project trains a Convolutional Neural Network (CNN) to recognize letter tiles from the game *Word Link*. The model classifies 26 letters (A–Z) plus the special "QU" digraph from 64×64 grayscale tile images.

---

## 1. Dataset Generation

**Script:** `generate_dataset.py`

### Base Crops
- Hand-cropped letter images are stored in **`base_crops/`** (24 letters) and **`synthetic_bases/`** (J, Q, Z — 3 letters).
- Each base crop is a high-resolution (640×640) PNG of a single letter on a white background.

### Preprocessing (shave & resize)
Each base crop is **shaved** to remove excess whitespace, then **resized** to 64×64:
- Default shave: 65 pixels from left/right, 65 from top, 10 from bottom.
- **R**: shaved 85px to eliminate corner-triangle artifacts.
- **QU**: shaved only 20px from the right to keep the *u* visible.

### Augmentation Pipeline
500 variants are generated per letter (800 for "hard" letters: **R, W, QU**) using random:
| Augmentation | Range (normal) | Range (hard) |
|---|---|---|
| Rotation | ±3° | ±8° |
| Shift (x/y) | ±10 px | ±10 px |
| Brightness | 0.7×–1.3× | 0.7×–1.3× |
| Contrast | 0.8×–1.4× | 0.8×–1.4× |

Synthetic letters **J, Q, Z** additionally have 0–5 random black dots drawn near the bottom of the tile (simulating game-board dots).

### Output
All variants are saved to **`dataset/<LETTER>/`** subdirectories, ready for TensorFlow's `image_dataset_from_directory`.

---

## 2. Model Architecture

**Script:** `train_model.py`

### CNN Design

| Layer | Type | Details |
|---|---|---|
| Input | Rescaling | 1/255 normalization, shape (64, 64, 1) |
| Block 1 | Conv2D + BatchNorm + MaxPool | 32 filters, 3×3, ReLU, pool 2×2 |
| Block 2 | Conv2D + BatchNorm + MaxPool | 64 filters, 3×3, ReLU, pool 2×2 |
| Block 3 | Conv2D + BatchNorm + MaxPool | 128 filters, 3×3, ReLU, pool 2×2 |
| Block 4 | Conv2D + BatchNorm | 128 filters, 3×3, ReLU |
| Classifier | Flatten → Dense → Dropout → Dense | 256 units, 0.5 dropout, softmax output |

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy

---

## 3. Training Process

**Script:** `train_model.py`

### Data Loading
- TensorFlow's `image_dataset_from_directory` loads **`dataset/`** in grayscale at 64×64.
- 80% training / 20% validation split (fixed seed 123).
- Pipeline optimized with **caching**, **shuffle (buffer=1000)**, and **prefetch (AUTOTUNE)**.

### Training
- **Epochs:** 30
- **Batch size:** 32
- Training + validation accuracy/loss are plotted at the end.

### Output
- Trained model saved as **`perfect_ocr_model.h5`** (Keras H5 format).

---

## 4. Inference / Testing

**Script:** `test.py`

- Loads `perfect_ocr_model.h5`, runs prediction on 16 tiles from the **`test/`** folder.
- Tiles are resized directly to 64×64 (no shaving — the CNN learned that internally).
- Predictions below a **45% confidence** threshold are marked as `?`.
- Output is displayed as a 4×4 board.

---

## 5. Project Structure

```
base_crops/        # Hand-cropped base letter images (24 letters)
synthetic_bases/   # Synthetic base crops (J, Q, Z)
dataset/           # Generated augmented variants (26 letter folders)
preview/           # Preview images for sanity-checking bases
test/              # 16 test tile images (4×4 board)
generate_dataset.py  # Step 1: generate augmented dataset
train_model.py       # Step 2: train the CNN
test.py              # Step 3: run inference on test tiles
make_qu.py           # Utility to generate the QU base crop
perfect_ocr_model.h5 # Trained model weights
```

---

## 6. How to Retrain

```bash
# 1. (Optional) Generate or update dataset
python generate_dataset.py

# 2. Train the model
python train_model.py

# 3. Test on tile images
python test.py
```
