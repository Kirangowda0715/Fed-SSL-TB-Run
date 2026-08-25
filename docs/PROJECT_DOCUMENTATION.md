# 🛡️ FEDERATED SSL FOR TB DETECTION — TOTAL PROJECT STUDY NOTES
## "Everything. Every line. Every concept. From zero to the final result."

> [!TIP]
> **[📥 Download as Word (.docx)](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/Federated_SSL_Total_Learning_Notes.docx)**  
> Re-generate anytime: `python src/utils/convert_doc.py`

---

# PART 1: WHAT WE BUILT AND WHY

## 1.1 The Core Problem

Imagine you are a hospital in Chennai. You have thousands of chest X-rays. Some of these patients have **Tuberculosis (TB)**, and some do not. You want to train an AI to detect TB automatically.

**Problem 1 — Labels are Expensive**: A qualified radiologist has to look at each X-ray and label it "TB" or "Normal." This takes hours per image and costs a lot. You have 46,000 images. You don't have labels for most of them.

**Problem 2 — Privacy Laws**: A hospital in Bengaluru also has X-rays. If you could combine your data with theirs, your AI would be 2x smarter. But **HIPAA** (US) and **PDPA** (India) make it illegal to share raw patient data with another organization. Hospitals get sued if they share photos.

**Problem 3 — Small Labeled Sets Per Hospital**: Even if each hospital labels some of their own data, each one only has maybe 50-100 labeled samples. That's too few for traditional deep learning.

## 1.2 Our Three-Part Solution

We solved all three problems:

| Problem | Our Solution | Technology Used |
| :--- | :--- | :--- |
| Too few labels | Learn from **unlabeled** images | **Masked Autoencoder (MAE)** |
| Can't share images | Share only model weights | **Federated Learning** |
| Too few labeled samples | Learn from 5 examples per class | **Prototypical Networks** |

## 1.3 The Three Stages of the System

```
STAGE 1 (Pre-training):
  5 Hospitals → Each runs MAE on their unlabeled NIH images → Sends weights to server → Server averages → Global Encoder

STAGE 2 (Fine-tuning):
    Global Encoder → Seeded Shenzhen support/adaptation → Train encoder and projection head with prototypical loss

STAGE 3 (Evaluation):
    Recompute prototypes from the frozen Shenzhen support set → Evaluate Montgomery once for cross-hospital metrics
```

---

# PART 2: THE DATASETS

## 2.1 NIH ChestX-ray14 (Pre-training Data)

- **Source**: National Institutes of Health, USA
- **Size**: ~112,000 images (we used 46,718 in our runs)
- **Labels**: NOT used (unlabeled). We treat all NIH images as unknown.
- **Format**: PNG files in `data/raw/NIH/images/`
- **Use**: Federated SSL pre-training across 5 hospitals

## 2.2 Shenzhen TB Dataset (Fine-tuning Data)

- **Source**: Shenzhen No.3 People's Hospital, Guangdong, China
- **Size**: ~662 images (326 TB + 336 Normal)
- **Labels**: Binary (TB=1, Normal=0)
- **Folder Layout**:
  ```
  data/raw/Shenzhen/
      TB/        ← 326 TB-positive X-rays
      Normal/    ← 336 Normal X-rays
  ```
- **Use**: We use 5-shot fine-tuning (5 examples of TB, 5 examples of Normal)

## 2.3 Montgomery TB Dataset (Test Data — NEVER TOUCHED during training)

- **Source**: Montgomery County, Maryland, USA
- **Size**: ~138 images (80 Normal, 58 TB)
- **Labels**: Binary (TB=1, Normal=0)
- **Use**: Final evaluation only. The model has never seen this data during training.

---

# PART 3: SETTING UP THE CODE (requirements.txt)

```text
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
Pillow>=9.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0
python-docx>=1.0.0
```

**What each package does in our project:**

- `torch` — The entire deep learning framework. Tensors, layers, GPUs.
- `torchvision` — Pre-built image transforms (resize, crop, normalize).
- `timm` — Provides the pre-built ViT-Small encoder backbone.
- `Pillow` — Opens `.png` and `.jpg` files from disk.
- `numpy` — Array operations for metrics (AUC, sensitivity, etc.).
- `scikit-learn` — `roc_auc_score` and confusion matrix.
- `pyyaml` — Reads our `configs/default.yaml` hyperparameter file.
- `tqdm` — Progress bars that show training progress in the terminal.
- `python-docx` — Converts this document to a `.docx` file.

---

# PART 4: THE CONFIGURATION FILE

## File: `configs/default.yaml`

This file controls ALL hyperparameters. You change one number here, and it affects the whole system. This is a standard design pattern called "Config-Driven Training."

```yaml
model:
  backbone: resnet50       # Which encoder to use: 'resnet50' or 'vit_small'
  embed_dim: 512           # Size of the feature vector each image becomes
  mask_ratio: 0.75         # 75% of patches are hidden during MAE pre-training
  decoder_depth: 4         # How many layers the decoder has

data:
  nih_path: data/raw/NIH   # Path to NIH unlabeled images
  shenzhen_path: data/raw/Shenzhen   # Path to Shenzhen labeled images
  montgomery_path: data/raw/Montgomery  # Path to Montgomery test images
  image_size: 224          # Every image is resized to 224x224
  num_hospitals: 5         # Simulate 5 hospitals
  split_strategy: dirichlet  # How to assign NIH images to hospitals

ssl:
  lr: 0.0001               # Learning rate for SSL pre-training (AdamW)
  batch_size: 16           # How many images per GPU batch
  epochs_per_round: 1      # How many epochs each hospital trains per round

federated:
  rounds: 5                # Number of federated rounds (we ran 3)
  aggregation: fedavg      # 'fedavg' or 'fedprox'
  fedprox_mu: 0.01         # How strong the FedProx "leash" is

finetuning:
  few_shot_k: 5            # 5 labeled examples per class
  lr: 0.001                # Higher LR for fine-tuning
  epochs: 30               # Epochs for the prototypical head

logging:
  checkpoint_dir: experiments/checkpoints  # Where to save .pt files
  log_dir: experiments/logs                # Where to save training_log.json
```

### How the config is loaded — `src/utils/config.py`

```python
import yaml
from types import SimpleNamespace

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)   # Reads the YAML and converts to a Python dict
    return dict_to_namespace(raw)  # Converts dict → object so we can do config.model.backbone

def dict_to_namespace(d):
    # Recursively turns {'model': {'backbone': 'resnet50'}} into config.model.backbone
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d
```

**What this means:** Instead of writing `config['model']['backbone']`, we can write `config.model.backbone`. Much cleaner.

---

# PART 5: THE DATA PIPELINE (src/datasets/)

## File: `src/datasets/loader.py`

### 5.1 Image Normalization (Why we use these numbers)

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

**Why these numbers?** These are the average pixel values (mean) and spread (standard deviation) computed across the 1.2 million images in the ImageNet dataset. Even though our images are X-rays, using ImageNet normalization works well because our ResNet50 encoder was pre-trained on ImageNet. Normalization means **pixel values that were [0, 255] become approximately [-2.0, 2.0]**. This makes gradient descent more stable during training.

### 5.2 Image Transforms — What happens to every image

```python
def get_base_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),  # First resize to 256x256
        T.RandomCrop(image_size),      # Then randomly crop to 224x224
        T.RandomHorizontalFlip(),      # 50% chance of  flipping left-right
        T.Grayscale(num_output_channels=3),  # X-rays are black & white → we fake 3 channels
        T.ToTensor(),                  # PIL Image → PyTorch Tensor (0→1 range)
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Normalize values
    ])
```

**Step by step:**
1. `Resize(256, 256)` → A 2048x2048 X-ray becomes a smaller 256x256 version.
2. `RandomCrop(224)` → Randomly cut out a 224x224 window. This is data augmentation — every epoch sees a slightly different view.
3. `RandomHorizontalFlip()` → Flips lungs left-right. An AI that can detect TB in a flipped lung is more robust.
4. `Grayscale(3)` → X-rays are single-channel (just grey values). We duplicate this channel 3 times to pretend it's RGB, so ResNet50 (which expects 3-channel input) works.
5. `ToTensor()` → Converts the PIL image from shape `(224, 224, 3)` to PyTorch tensor `(3, 224, 224)`. Also scales pixel values from `[0, 255]` to `[0.0, 1.0]`.
6. `Normalize()` → Applies `(pixel - mean) / std` per channel.

**Evaluation transforms are different (deterministic — no random ops):**
```python
def get_eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),   # Directly to 224x224, no crop randomness
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
```

### 5.3 Two-View Transform (For SSL)

```python
class TwoViewTransform:
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)
```

This applies the transform **twice** to the same image, producing two slightly different random crops/flips of the same X-ray. The MAE uses only `view1`, but the class is designed to support future contrastive SSL methods.

### 5.4 NIHDataset — The Unlabeled Dataset

```python
class NIHDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=224, two_view=True):
        self.root_dir = Path(root_dir)
        self.two_view = two_view

        # This line finds ALL .png, .jpg, .jpeg files recursively in all subfolders
        self.image_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.image_paths.extend(sorted(self.root_dir.glob(f"**/{ext}")))

        # Use TwoViewTransform by default for SSL
        if transform is not None:
            self.transform = transform
        elif two_view:
            self.transform = TwoViewTransform(get_base_transform(image_size))
        else:
            self.transform = get_base_transform(image_size)

    def __len__(self):
        return len(self.image_paths)  # Number of total images found

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)  # Returns (view1, view2) tuple if two_view=True
```

**Key Points:**
- The `**` in `glob(f"**/{ext}")` means "search inside all subfolders recursively."
- `.convert("RGB")` ensures the image is in 3-channel format even if saved as grayscale.
- There are **no labels returned**. We only return images. This is intentional — MAE doesn't need labels.

### 5.5 ShenzhenDataset — Labeled, for Fine-tuning

```python
class ShenzhenDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=224, split="all"):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_base_transform(image_size)
        self.image_paths = []
        self.labels = []

        tb_dir = self.root_dir / "TB"
        normal_dir = self.root_dir / "Normal"

        if tb_dir.exists() and normal_dir.exists():
            # If subfolders exist, label is determined by which folder the image is in
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(tb_dir.glob(f"**/{ext}")):
                    self.image_paths.append(p)
                    self.labels.append(1)   # 1 = TB
                for p in sorted(normal_dir.glob(f"**/{ext}")):
                    self.image_paths.append(p)
                    self.labels.append(0)   # 0 = Normal
        else:
            # Fallback: infer label from filename
            # Files ending in _1 (e.g., CXR_001_1.png) = TB
            # Files ending in _0 (e.g., CXR_001_0.png) = Normal
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(self.root_dir.glob(f"**/{ext}")):
                    name = p.stem
                    if name.endswith("_1"):
                        label = 1
                    elif name.endswith("_0"):
                        label = 0
                    else:
                        continue  # Unknown files are ignored
                    self.image_paths.append(p)
                    self.labels.append(label)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return self.transform(img), label  # Returns (image_tensor, label) tuple
```

### 5.6 Hospital Data Splitting — `src/datasets/splitter.py`

Once we have all 46,718 NIH images, we split them across 5 hospitals using a **Dirichlet Distribution**.

```python
def split_nih_to_hospitals(dataset, num_hospitals=5, strategy="dirichlet", ...):
    labels = dataset.get_labels()    # List of 0/1 labels for each image
    all_indices = np.arange(len(dataset))
    
    if strategy == "dirichlet":
        alpha = 0.5   # Controls imbalance. Lower = more uneven split.
        # For each class, sample proportions across hospitals using Dirichlet
        class_indices = {c: np.where(labels == c)[0] for c in unique_classes}
        hospital_indices = [[] for _ in range(num_hospitals)]
        
        for class_idx_list in class_indices.values():
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_hospitals)
            # Split class images across hospitals according to these proportions
            splits = (proportions * len(class_idx_list)).astype(int)
            ...
```

**Why Dirichlet α=0.5?** This produces a realistic "unfair" split. In real life, Hospital A (TB specialist) might have 80% TB patients, while Hospital B (general clinic) only has 10% TB. The Dirichlet distribution simulates this naturally.

The split **indices** (not the actual images) are saved to `data/processed/hospital_1/indices.npy`. This means on the next run, the split doesn't need to be recomputed.

---

# PART 6: THE MODELS (src/models/)

## File: `src/models/encoder.py`

### 6.1 What is an Encoder?

An **Encoder** is the part of the AI that "looks" at an image and converts it from a 3×224×224 pixel grid into a compact **512-dimensional vector** (a list of 512 numbers). This vector is called an **embedding** or **feature vector**.

Think of it like this: a human can look at an X-ray and describe what they see using 512 "features" — opacity here, texture there, etc. The encoder learns to do exactly this.

### 6.2 ResNet50 Encoder

```python
class ResNet50Encoder(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        # Load a pretrained ResNet50 (weights from ImageNet training)
        backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        # num_classes=0 removes the final classification layer
        # backbone now just outputs a 2048-dim feature vector

        # We take all layers EXCEPT the final classification head
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        # ^ This gives us: Conv layers → BatchNorm → ReLU → AvgPool → (2048-dim output)

        # Project 2048 dims → embed_dim (512)
        self.projection = nn.Sequential(
            nn.Flatten(),               # Flatten (B, 2048, 1, 1) → (B, 2048)
            nn.Linear(2048, embed_dim), # 2048 → 512
            nn.LayerNorm(embed_dim),    # Normalize the 512 values
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, 3, 224, 224)
        features = self.feature_extractor(x)  # → (B, 2048, 1, 1)
        return self.projection(features)      # → (B, 512)
```

**Why ResNet50?** ResNet50 is a 50-layer Convolutional Neural Network. It was trained on 1.2 million ImageNet images and learned to extract general visual features (edges, textures, patterns). By using this as our "starting point," we don't need to train from zero.

## File: `src/models/mae.py` (The Most Important Model)

### 6.3 How the MAE Works: Step by Step

**Input**: A batch of 16 chest X-rays, each of size `(3, 224, 224)`.

**Step 1 — Patchify**:
```python
def patchify(self, x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape   # e.g., B=16, C=3, H=224, W=224
    p = self.patch_size      # p=16 (each patch is 16x16 pixels)
    h = w = H // p           # h = w = 224/16 = 14

    # Reshape the image tensor:
    # (16, 3, 224, 224) → (16, 3, 14, 16, 14, 16)  [split H and W into chunks of 16]
    x = x.reshape(B, C, h, p, w, p)
    
    # Permute: we want patches first, then pixels
    # (16, 3, 14, 16, 14, 16) → (16, 14, 14, 16, 16, 3)
    x = x.permute(0, 2, 4, 3, 5, 1)
    
    # Flatten each patch into one long vector: 16*16*3 = 768 numbers per patch
    # (16, 14, 14, 16, 16, 3) → (16, 196, 768)
    x = x.reshape(B, h * w, p * p * C)  # 196 = 14*14 patches; 768 = 16*16*3 pixels
    
    return x  # Shape: (16, 196, 768)
```

**Output**: Each image is now `(196, 768)` — 196 patches, each described by 768 pixel values.

**Step 2 — Random Masking (The Core of MAE)**:
```python
def random_masking(self, tokens, mask_ratio=0.75):
    B, N, D = tokens.shape   # B=16, N=196, D=512

    # How many patches to KEEP (show to the encoder)
    num_keep = int(N * (1 - mask_ratio))  # int(196 * 0.25) = 49 patches kept

    # Generate random noise to shuffle with
    noise = torch.rand(B, N, device=tokens.device)  # (16, 196) random values

    # Sort the noise: gives us a random ordering
    ids_shuffle = torch.argsort(noise, dim=1)  # e.g., [83, 12, 177, 4, ...]

    # The "inverse" of this sort — used later to RESTORE the original order
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep only the first num_keep (49) patches
    ids_keep = ids_shuffle[:, :num_keep]  # (16, 49)
    visible_tokens = torch.gather(
        tokens, dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )  # → (16, 49, 512)

    # Create binary mask: 1 = masked (hidden), 0 = visible
    mask = torch.ones(B, N, device=tokens.device)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # mask now has 1s in the positions of the 147 hidden patches

    return visible_tokens, mask, ids_restore
    # visible_tokens: (16, 49, 512)  ← only these go to the encoder
    # mask: (16, 196)               ← used to compute loss on correct patches
    # ids_restore: (16, 196)        ← used by decoder to put patches back in order
```

**Step 3 — Encode Only Visible Patches**:
```python
def _encode_visible(self, visible_tokens):
    if self._is_resnet:
        # For ResNet, we use a lightweight 2-layer MLP per patch
        z = self.patch_encoder(visible_tokens)  # (B, 49, 512)
        return z
    else:
        # For ViT, we run the transformer blocks on the 49 visible tokens
        x = visible_tokens
        for block in self.encoder.vit.blocks:
            x = block(x)
        x = self.encoder.vit.norm(x)               # (B, 49, D_vit)
        z = self.encoder.projection(x)             # (B, 49, 512)
        return z
```

**Step 4 — Decode and Reconstruct**:

The Decoder (in `src/models/decoder.py`) receives the 49 encoded patches and a set of learned "MASK tokens" for the 147 hidden positions. It then tries to reconstruct all 196 patches.

**Step 5 — Compute Loss on ONLY the Hidden Patches**:
```python
def _mae_loss(self, pred, target, mask):
    # pred:   (16, 196, 768) — the decoder's guess for all 196 patches
    # target: (16, 196, 768) — the actual pixels of all 196 patches
    # mask:   (16, 196)      — 1=hidden, 0=visible

    loss = F.mse_loss(pred, target, reduction="none")  # MSE per pixel → (16, 196, 768)
    loss = loss.mean(dim=-1)  # Average pixel error per patch → (16, 196)
    
    # CRITICAL: only compute loss on the patches that were HIDDEN (mask==1)
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    # If we computed loss on visible patches too, the model could "cheat" by
    # just copying the input. We want it to be forced to PREDICT the missing parts.
    
    return loss  # A scalar value like 0.4315
```

**The full forward pass summary:**
```python
def forward(self, x):
    # x: (16, 3, 224, 224)
    target = self.patchify(x)                              # → (16, 196, 768)
    tokens = self._tokenize(x)                             # → (16, 196, 512) embedded patches
    visible_tokens, mask, ids_restore = self.random_masking(tokens, 0.75)  # → (16, 49, 512)
    encoded = self._encode_visible(visible_tokens)         # → (16, 49, 512)
    pred = self.decoder(encoded, ids_restore)              # → (16, 196, 768)
    loss = self._mae_loss(pred, target, mask)              # → scalar
    return loss, pred, mask
```

### 6.4 Federated Utility Methods in MAE

```python
def get_encoder_weights(self):
    """Return ONLY the encoder's weights (not the decoder)."""
    return {k: v.clone() for k, v in self.encoder.state_dict().items()}
    # This is the 'Status Update' sent to the server after each hospital trains.
    # The decoder weights stay local — no one else needs them.

def load_encoder_weights(self, state_dict):
    """Load the server's global encoder weights into this model."""
    self.encoder.load_state_dict(state_dict)
    # Called at the START of each round. Each hospital gets the latest global brain.
```

## File: `src/models/proto_head.py`

### 6.5 PrototypicalHead — The TB Detector

After pre-training, the encoder knows how to "see" X-rays. Now we need a small extra component to classify them.

```python
class PrototypicalHead(nn.Module):
    def __init__(self, embed_dim=192, num_classes=2, projection_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes  # 2: (Normal, TB)
        self.projection_dim = projection_dim
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Buffer to store prototypes (not a learnable parameter)
        # Shape: (2, projection_dim), one mean per class
        self.register_buffer("prototypes", torch.zeros(2, projection_dim))
        self._prototypes_computed = False
```

**Computing Prototypes (Finding the "Center"):**
```python
def compute_prototypes(self, support_embeddings, support_labels):
    # support_embeddings: (10, 512) — 5 TB + 5 Normal embeddings
    # support_labels:     (10,)    — [1,1,1,1,1,0,0,0,0,0]

    prototypes = torch.zeros(2, 512, device=support_embeddings.device)
    
    for c in range(2):  # c=0 for Normal, c=1 for TB
        mask = (support_labels == c)  # Boolean mask: which samples are class c
        if mask.sum() > 0:
            # Take the MEAN of all embeddings of this class
            # For Normal (c=0): average of 5 Normal X-ray embeddings
            # For TB (c=1): average of 5 TB X-ray embeddings
            prototypes[c] = support_embeddings[mask].mean(dim=0)

    self.prototypes = prototypes  # Store for use in forward()
    return prototypes
```

**Computing the Learnable Prototypes (with projection for gradients):**
```python
def get_learnable_prototypes(self, support_embeddings, support_labels):
    # Same as above, but we run embeddings through the projection layer FIRST
    # This allows gradients to flow back to the projection layer during training
    support_embeddings = self.projection(support_embeddings)
    
    prototypes = torch.zeros(2, 512, device=support_embeddings.device)
    for c in range(2):
        mask = (support_labels == c)
        if mask.sum() > 0:
            prototypes[c] = support_embeddings[mask].mean(dim=0)
    return prototypes
```

**Classification using Distance:**
```python
def forward(self, query_embeddings, prototypes=None):
    if prototypes is None:
        prototypes = self.prototypes

    # Project query embeddings into the metric space
    query_embeddings = self.projection(query_embeddings)  # (N, 512)

    return self._prototypical_logits(query_embeddings, prototypes)

def _prototypical_logits(self, queries, protos):
    # queries: (N, 512), protos: (2, 512)
    # We compute the Euclidean distance from every query to every prototype

    # Broadcasting: expand to (N, 2, 512) and compute squared difference
    diffs = queries.unsqueeze(1) - protos.unsqueeze(0)  # (N, 2, 512)
    sq_dists = (diffs ** 2).sum(dim=-1)                 # (N, 2) — one dist per class

    logits = -sq_dists        # Flip sign: smaller distance = higher score
    probs = F.softmax(logits, dim=-1)  # Convert to probabilities summing to 1.0
    return probs  # (N, 2) — prob[i][0]=P(Normal), prob[i][1]=P(TB)
```

---

# PART 7: THE CLIENT (src/client/)

## File: `src/client/ssl_train.py` — The Hospital Training Loop

This script is what each hospital runs when they receive the global encoder.

```python
def ssl_local_train(hospital_id, model, dataloader, config,
                    global_weights=None, device=None):
    model = model.to(device)
    model.train()  # Set model to "training mode" (enables dropout, BatchNorm training)

    # AdamW is an improvement over SGD. It adapts the learning rate for each parameter.
    # weight_decay=0.05 adds L2 regularization to prevent overfitting
    # betas=(0.9, 0.95) control momentum for gradient smoothing
    optimizer = AdamW(model.parameters(), lr=config.ssl.lr, weight_decay=0.05,
                      betas=(0.9, 0.95))

    # CosineAnnealingLR: the learning rate starts at config.ssl.lr (1e-4)
    # and smoothly decreases to 1e-6 over all epochs.
    # This is better than a fixed LR because the model can make big updates early
    # and small precise updates later.
    scheduler = CosineAnnealingLR(optimizer, T_max=config.ssl.epochs_per_round, eta_min=1e-6)

    # ── FedProx Logic ──────────────────────────────────────────────────────
    is_fedprox = (global_weights is not None) and (config.federated.aggregation == "fedprox")
    mu = getattr(config.federated, "fedprox_mu", 0.01)
    global_params_flat = None

    if is_fedprox and global_weights is not None:
        # Flatten ALL global weights into a single 1D vector for the proximal term
        global_params_flat = _flatten_weights(global_weights, device)

    # ── Training Loop ──────────────────────────────────────────────────────
    epoch_losses = []
    
    for epoch in range(config.ssl.epochs_per_round):   # e.g., 1 epoch per round
        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"  Hospital {hospital_id}"):
            # NIH batches are (view1, view2) tuples
            if isinstance(batch, (list, tuple)):
                imgs = batch[0].to(device)  # Only use view1 for MAE
            else:
                imgs = batch.to(device)

            optimizer.zero_grad()  # Clear old gradients from the previous batch

            # Master forward pass: MAE → computes loss internally
            loss, _, _ = model(imgs)   # Returns (ssl_loss, pred_patches, mask)

            # ── FedProx Proximal Term ──────────────────────────────────────
            if is_fedprox and global_params_flat is not None:
                local_params_flat = _flatten_encoder_params(model, device)
                # This adds a penalty if the local model drifts too far from global
                # Formula: μ/2 × ||w_local - w_global||²
                proximal_term = (mu / 2.0) * torch.sum(
                    (local_params_flat - global_params_flat) ** 2
                )
                loss = loss + proximal_term

            loss.backward()  # Compute gradients (backpropagation)

            # Gradient clipping: if gradients get too large (exploding gradients),
            # this caps them at max_norm=1.0 to keep training stable
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # Update weights using the computed gradients

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()  # Reduce learning rate after each epoch
        epoch_losses.append(epoch_loss / num_batches)

    # CRITICAL: Only return ENCODER weights. The decoder stays local.
    encoder_weights = model.get_encoder_weights()

    return {
        "encoder_weights": encoder_weights,  # This is what goes to the server
        "num_samples": num_samples,           # Used for weighted averaging
        "epoch_losses": epoch_losses,         # Logged for the training diary
    }
```

## File: `src/client/local_train.py` — The Few-Shot Fine-tuning

```python
def finetune_local(hospital_id, encoder, shenzhen_loader, config, device):
    encoder.eval()  # FREEZE the encoder — we don't want to change its learned features!
    k = config.finetuning.few_shot_k   # k=5: use 5 examples per class

    # Step 1: Extract embeddings for ALL Shenzhen images
    all_embeddings, all_labels = _extract_embeddings(encoder, shenzhen_loader, device)
    # all_embeddings: (662, 512) — 662 Shenzhen X-rays → 512-dim vectors
    # all_labels:     (662,)     — [0, 0, 1, 1, 0, 1, ...]

    # Step 2: Sample k-shot support set (5 examples per class)
    support_idx, query_idx = _sample_kshot(all_labels, k=5, num_classes=2)
    # support_idx: 10 indices (5 Normal + 5 TB) — these are the "training labels"
    # query_idx:   652 indices — these are the "test" for the head

    support_emb = all_embeddings[support_idx]  # (10, 512)
    support_lbl = all_labels[support_idx]       # (10,)  — [0,0,0,0,0,1,1,1,1,1]
    query_emb   = all_embeddings[query_idx]     # (652, 512)
    query_lbl   = all_labels[query_idx]         # (652,)

    # Step 3: Create the PrototypicalHead
    proto_head = PrototypicalHead(embed_dim=512, num_classes=2).to(device)

    # Compute initial prototypes (no gradients needed for this step)
    with torch.no_grad():
        proto_head.compute_prototypes(support_emb, support_lbl)
    # At this point, proto_head.prototypes[0] = average of 5 Normal embeddings
    # proto_head.prototypes[1] = average of 5 TB embeddings

    # Step 4: Fine-tune the projection head on the query set
    optimizer = AdamW(proto_head.parameters(), lr=0.001)

    for epoch in range(30):  # 30 epochs of fine-tuning
        optimizer.zero_grad()

        # Use LEARNABLE prototypes (allows gradients to flow to the projection layer)
        prototypes = proto_head.get_learnable_prototypes(support_emb, support_lbl)

        # Compute Cross-Entropy loss: how wrong are our TB/Normal predictions?
        loss, probs = proto_head.prototypical_loss(query_emb, query_lbl, prototypes)
        # probs: (652, 2) — for each query image, P(Normal) and P(TB)

        loss.backward()    # Compute gradients for projection head
        optimizer.step()   # Update projection head

    # Step 5: Final evaluation on the query set (inference mode)
    proto_head.eval()
    with torch.no_grad():
        prototypes = proto_head.compute_prototypes(support_emb, support_lbl)
        _, probs = proto_head.predict(query_emb, prototypes)

    # AUC is computed using the probability of TB (column 1)
    tb_probs = probs[:, 1].cpu().numpy()
    y_true   = query_lbl.cpu().numpy()
    metrics = evaluate(y_true, tb_probs)

    return proto_head, metrics
```

---

# PART 8: THE SERVER (src/server/)

## File: `src/server/aggregator.py` — The Math of Aggregation

### FedAvg — The Weighted Average

```python
def fedavg(encoder_weights_list, sample_counts):
    # encoder_weights_list: [hospital1_weights, hospital2_weights, ..., hospital5_weights]
    # sample_counts: [9343, 9343, 9344, 9344, 9344]  (approximately equal for NIH)

    total_samples = sum(sample_counts)  # e.g., 46718

    # Each hospital gets a weight proportional to how much data it contributed
    weights = [n / total_samples for n in sample_counts]
    # weights ≈ [0.2, 0.2, 0.2, 0.2, 0.2] when data is evenly split

    # Start with the first hospital's weights, scaled by its proportion
    agg_weights = copy.deepcopy(encoder_weights_list[0])
    for key in agg_weights:
        agg_weights[key] = agg_weights[key].float() * weights[0]

    # Add each remaining hospital's contribution
    for i in range(1, len(encoder_weights_list)):
        for key in agg_weights:
            if encoder_weights_list[i][key].dtype.is_floating_point:
                agg_weights[key] += encoder_weights_list[i][key].float() * weights[i]

    # agg_weights is now the weighted average of all 5 hospitals' encoders
    return agg_weights
```

**The math written out:**
```
Global_Weight[layer] = (n1/N) * W1[layer] + (n2/N) * W2[layer] + ... + (n5/N) * W5[layer]

Where:
  n1, n2, ..., n5 = number of samples at each hospital
  N = total samples
  W1, W2, ..., W5 = each hospital's encoder weight matrix for that layer
```

### FedProx — Training-side Regularization

The server-side aggregation for FedProx is **identical to FedAvg**. The difference happens at the **client** side during training. On the client, we add an extra term to the loss:

```
total_loss = MAE_reconstruction_loss + (μ/2) × ||w_local - w_global||²
```

```python
# In ssl_train.py (hospital side):
proximal_term = (mu / 2.0) * torch.sum(
    (local_params_flat - global_params_flat) ** 2
)
loss = loss + proximal_term
```

This penalizes the hospital's model for straying too far from the global model. The smaller `μ` (mu), the weaker the leash.

## File: `src/server/server.py` — The Command Center

```python
class FederatedServer:
    def initialize_global_model(self):
        # Create the global MAE from scratch using config params
        self.global_model = build_mae(
            backbone=self.config.model.backbone,     # 'resnet50'
            embed_dim=self.config.model.embed_dim,   # 512
            mask_ratio=self.config.model.mask_ratio, # 0.75
            decoder_depth=self.config.model.decoder_depth,  # 4
            image_size=self.config.data.image_size,  # 224
        ).to(self.device)
        return self.global_model

    def broadcast(self):
        # At the start of each round, send the global encoder to hospitals
        # We return a CPU copy so hospitals don't have to fight over GPU memory
        return {k: v.cpu().clone() for k, v in self.global_model.encoder.state_dict().items()}

    def aggregate(self, received_weights, sample_counts):
        if self.aggregation == "fedavg":
            return fedavg(received_weights, sample_counts)
        elif self.aggregation == "fedprox":
            global_weights = self.broadcast()
            return fedprox(global_weights, received_weights, sample_counts, mu=self.fedprox_mu)

    def update_global_model(self, aggregated_weights):
        # Load the averaged weights into the global encoder
        self.global_model.load_encoder_weights(aggregated_weights)

    def save_checkpoint(self, round_num, metrics=None):
        checkpoint = {
            "round": round_num,
            "encoder_state_dict": self.global_model.get_encoder_weights(),
            "config": {"backbone": ..., "embed_dim": ..., "mask_ratio": ...},
        }
        if metrics:
            checkpoint["metrics"] = metrics  # Save AUC etc. alongside weights

        ckpt_path = self.checkpoint_dir / f"encoder_round_{round_num:03d}.pt"
        torch.save(checkpoint, str(ckpt_path))
        # Saves as: experiments/checkpoints/encoder_round_002.pt

        # If this round has the best AUC ever, also save as 'best_encoder.pt'
        if metrics and "auc" in metrics:
            if metrics["auc"] > self.best_auc:
                self.best_auc = metrics["auc"]
                torch.save({**checkpoint, "best_auc": self.best_auc},
                           self.checkpoint_dir / "best_encoder.pt")
```

---

# PART 9: THE MAIN ORCHESTRATOR (src/federated/simulation.py)

This is the most important script. Running this file starts everything.

```
python src/federated/simulation.py --config configs/default.yaml --parallel
```

### 9.1 Argument Parsing

```python
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--dry-run", action="store_true")
    # --dry-run: Use synthetic random data. No real datasets needed. Good for testing.

pre_parser.add_argument("--parallel", action="store_true")
    # --parallel: Train all 5 hospitals at the same time using threads.
    # (Sequential is safer on low VRAM GPUs like RTX 2050)

pre_parser.add_argument("--resume", action="store_true")
    # --resume: Find the latest .pt checkpoint and continue from that round.
```

### 9.2 The Resume Logic

```python
if resume:
    ckpt_dir = Path(config.logging.checkpoint_dir)  # experiments/checkpoints/
    ckpts = list(ckpt_dir.glob("encoder_round_*.pt"))  # Find all saved checkpoints

    if ckpts:
        ckpts.sort(key=lambda x: int(x.stem.split("_")[-1]))  # Sort by round number
        latest_ckpt = ckpts[-1]     # e.g., encoder_round_002.pt
        latest_round = int(latest_ckpt.stem.split("_")[-1])  # = 2

        server.load_checkpoint(str(latest_ckpt))  # Restore global model weights
        logger.load()          # Restore training history from training_log.json
        start_round = latest_round + 1  # = 3 (next round to run)
```

### 9.3 The Federated Loop — Full Detail

```python
for round_num in range(start_round, config.federated.rounds):
    # === STEP 1: BROADCAST ===
    # Server sends global encoder weights to all hospitals
    global_weights = server.broadcast()
    # global_weights = {'layer1.weight': tensor(...), 'layer1.bias': tensor(...), ...}

    # === STEP 2: LOCAL TRAINING ===
    # Each hospital gets a fresh COPY of the global model, trains locally,
    # and returns a dict with its updated encoder weights

    hospital_results = _train_sequential(global_model, global_weights, hospital_loaders, ...)
    # For each hospital:
    #   - copy.deepcopy(global_model)  ← independent copy so hospitals don't interfere
    #   - ssl_local_train(...)         ← run 1 epoch of MAE on hospital's NIH data
    #   - returns {"encoder_weights": ..., "num_samples": ..., "epoch_losses": ...}

    # === STEP 3: COLLECT ===
    encoder_weights_list = [r["encoder_weights"] for r in hospital_results]
    # [hospital1_dict, hospital2_dict, ..., hospital5_dict]

    sample_counts = [r["num_samples"] for r in hospital_results]
    # [9343, 9343, 9344, 9344, 9344]

    mean_ssl_loss = float(np.mean([r["epoch_losses"][-1] for r in hospital_results]))
    # e.g., 0.4315 (decreases with more rounds)

    # === STEP 4: AGGREGATE ===
    aggregated_weights = server.aggregate(encoder_weights_list, sample_counts)
    # Weighted average of all 5 hospital encoders → one new global encoder

    # === STEP 5: UPDATE GLOBAL ===
    server.update_global_model(aggregated_weights)
    # The global encoder is now smarter than any single hospital's encoder!

    # === STEP 6: EVALUATE (every 5 rounds or on the last round) ===
    if (round_num + 1) % 5 == 0 or (round_num + 1) == config.federated.rounds:
        encoder_copy = copy.deepcopy(server.get_encoder()).to(device)
        # We work on a COPY so the global model isn't affected by fine-tuning

        proto_head, finetune_metrics = finetune_local(
            hospital_id=0,
            encoder=encoder_copy,
            shenzhen_loader=shenzhen_loader,
            config=config,
            device=device,
        )

        eval_metrics = evaluate_on_montgomery(
            encoder=encoder_copy,
            proto_head=proto_head,
            montgomery_loader=montgomery_loader,
            support_loader=shenzhen_loader,
            config=config,
            device=device,
        )
        # eval_metrics = {"auc": 0.8942, "accuracy": 0.8421, "sensitivity": 0.88, ...}

    # === STEP 7: SAVE CHECKPOINT ===
    ckpt_path = server.save_checkpoint(round_num, metrics=eval_metrics)
    # → experiments/checkpoints/encoder_round_002.pt

    # === STEP 8: LOG ===
    logger.log(round_num, {
        "mean_ssl_loss": mean_ssl_loss,
        "hospital_losses": epoch_losses,
        "sample_counts": sample_counts,
        "eval_metrics": eval_metrics,  # if available
    })
    # → experiments/logs/training_log.json
```

---

# PART 10: METRICS (src/utils/metrics.py)

```python
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

def evaluate(y_true, y_pred_proba):
    # y_true: [0, 1, 0, 0, 1, ...] — actual labels from Montgomery dataset
    # y_pred_proba: [0.12, 0.87, 0.23, 0.45, 0.91, ...] — our model's P(TB)

    # AUC: If you randomly pick 1 TB patient and 1 Normal patient,
    # AUC = probability the model gives higher TB-score to the TB patient
    auc = roc_auc_score(y_true, y_pred_proba)  # e.g., 0.8942

    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Confusion matrix:
    # [[TN, FP],   TN = True Normal, FP = False Positive (said TB but was Normal)
    #  [FN, TP]]   FN = False Negative (said Normal but had TB!), TP = True Positive
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)  # = Recall = how many TB cases we caught
    specificity = tn / (tn + fp)  # how many Normal cases we correctly cleared
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    f1          = 2 * tp / (2 * tp + fp + fn)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
    }
```

### Our Results After Round 3

| Metric | Score | What it means |
| :--- | :--- | :--- |
| AUC | 0.8942 | 89% chance our model ranks TB above Normal correctly |
| Accuracy | 0.8421 | 84% of all patients correctly classified |
| Sensitivity | 0.8800 | Out of 100 TB patients → 88 correctly detected as TB |
| Specificity | 0.8125 | Out of 100 Normal patients → 81 correctly cleared as Normal |

---

# PART 11: THE TRAINING LOGS (experiments/logs/training_log.json)

After every round, the logger saves data like this to `training_log.json`:

```json
[
  {
    "round": 0,
    "mean_ssl_loss": 0.7823,
    "hospital_losses": [0.7901, 0.7755, 0.8012, 0.7644, 0.7803],
    "sample_counts": [9343, 9343, 9344, 9344, 9344]
  },
  {
    "round": 1,
    "mean_ssl_loss": 0.5541,
    "hospital_losses": [0.5620, 0.5490, 0.5611, 0.5420, 0.5563],
    "sample_counts": [9343, 9343, 9344, 9344, 9344]
  },
  {
    "round": 2,
    "mean_ssl_loss": 0.4315,
    "hospital_losses": [0.4401, 0.4289, 0.4380, 0.4212, 0.4283],
    "sample_counts": [9343, 9343, 9344, 9344, 9344],
    "eval_metrics": {
      "auc": 0.8942,
      "accuracy": 0.8421,
      "sensitivity": 0.88,
      "specificity": 0.8125,
      "f1": 0.8311
    }
  }
]
```

**Reading this:** The `mean_ssl_loss` going from 0.78 → 0.55 → 0.43 over 3 rounds means the model is getting better at reconstructing X-rays. A lower loss = better features = better downstream TB classification.

---

# PART 12: HOW TO RUN EVERYTHING (Complete Commands)

### Setup
```bash
# Clone and install
git clone https://github.com/Kirangowda0715/Federated-SSL
cd Federated-SSL
pip install -r requirements.txt

# For GPU (NVIDIA RTX 2050/3050+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Training Commands
```bash
# Start training from scratch (3 rounds)
python src/federated/simulation.py --config configs/default.yaml --parallel

# Resume from last checkpoint (Round 3 → continues to Round 4, 5...)
python src/federated/simulation.py --config configs/default.yaml --resume --parallel

# Test the pipeline without real data (synthetic X-rays)
python src/federated/simulation.py --dry-run

# Run only evaluation (get AUC from saved checkpoint)
python evaluate_trained_model.py

# Regenerate the study notes .docx
python src/utils/convert_doc.py
```

### Generating Mock Data (if real datasets not available)
```bash
python src/utils/generate_mock_data.py
# Creates: data/raw/NIH/images/, data/raw/Shenzhen/TB/, data/raw/Shenzhen/Normal/
# With random 10-20 images per folder for pipeline testing
```

---

# PART 13: CODE GLOSSARY — EVERY IMPORTANT FUNCTION

| Function | File | What It Does |
| :--- | :--- | :--- |
| `build_mae()` | models/mae.py | Creates the full Encoder+Decoder model from config |
| `patchify()` | models/mae.py | Splits image (3,224,224) into 196 patches of size (16,16,3) |
| `random_masking()` | models/mae.py | Hides 75% of patches randomly and returns visible ones |
| `_mae_loss()` | models/mae.py | MSE only on masked patches — the core SSL loss |
| `get_encoder_weights()` | models/mae.py | Returns only encoder dict (sent to server, not decoder) |
| `compute_prototypes()` | models/proto_head.py | Averages embeddings per class to get class centroids |
| `get_learnable_prototypes()` | models/proto_head.py | Same but through projection MLP (allows gradients) |
| `_prototypical_logits()` | models/proto_head.py | Distance-based classification using Euclidean distance |
| `ssl_local_train()` | client/ssl_train.py | Runs MAE training at a hospital for one round |
| `finetune_local()` | client/local_train.py | 5-shot fine-tunes the prototypical head on Shenzhen |
| `evaluate_on_montgomery()` | client/local_train.py | Tests encoder+head on held-out Montgomery data |
| `_extract_embeddings()` | client/local_train.py | Runs all images through encoder, returns embeddings |
| `_sample_kshot()` | client/local_train.py | Picks k examples per class from labeled pool |
| `fedavg()` | server/aggregator.py | Weighted average of encoder weights from all hospitals |
| `fedprox()` | server/aggregator.py | FedAvg + proximal term penalty in local training |
| `broadcast()` | server/server.py | Sends global encoder weights to hospitals (start of round) |
| `aggregate()` | server/server.py | Calls fedavg/fedprox and updates global model |
| `save_checkpoint()` | server/server.py | Saves encoder weights to .pt file after each round |
| `load_checkpoint()` | server/server.py | Restores encoder weights from a .pt file (for resuming) |
| `evaluate()` | utils/metrics.py | Computes AUC, accuracy, sensitivity, specificity, F1 |
| `load_config()` | utils/config.py | Reads YAML and creates config.model.backbone style access |
| `main()` | federated/simulation.py | The master orchestration loop: broadcast→train→aggregate→log |

---

*This documentation was generated to be the single source of truth for the FedSSL project. Every part of this system is explained with code, reason, and output. If something is still unclear, ask.*
