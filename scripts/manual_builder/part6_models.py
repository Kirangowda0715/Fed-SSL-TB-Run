"""
scripts/manual_builder/part6_models.py
--------------------------------------
Part VI: Mathematical and PyTorch Reconstruction of Vision Transformers,
Masked Autoencoders, and Optimization Mechanics.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part6(doc):
    add_h1(doc, "Part VI: Mathematical & PyTorch Reconstruction of Models")

    add_h2(doc, "6.1 Vision Transformer (ViT) Foundations from First Principles")
    add_body(doc, "Traditional Convolutional Neural Networks (CNNs) process images via local inductive biases (spatial locality and translation invariance). The Vision Transformer (ViT; Dosovitskiy et al., ICLR 2021) discards convolutions, treating an image as a sequence of discrete visual tokens and learning global relationships via Multi-Head Self-Attention.")

    add_h3(doc, "1. Patch Extraction & Linear Projection")
    add_body(doc, "Given a 2D chest radiograph x in R^(H x W x C) with H = W = 224 and C = 3, we partition the image into non-overlapping spatial patches of size P x P (where P = 16):")

    add_bullet(doc, "N = (H / P) * (W / P) = (224 / 16) * (224 / 16) = 14 * 14 = 196 patches.", bold_prefix="Number of Patches: ")
    add_bullet(doc, "Each 16x16x3 patch is flattened into a 1D pixel vector x_p in R^(P^2 * C) = R^768.", bold_prefix="Raw Patch Dimension: ")
    add_bullet(doc, "A trainable linear projection matrix E in R^(768 x D) maps each patch vector into the transformer latent embedding space of dimension D = 192 (for ViT-Tiny): z_0 = [x_p^1 E; x_p^2 E; ...; x_p^N E] in R^(N x D).", bold_prefix="Linear Patch Embedding: ")

    add_h3(doc, "2. Positional Encoding")
    add_body(doc, "Because transformer attention operations are permutation-invariant (they have no inherent sense of 2D geometry), spatial coordinates must be injected into the patch sequence. Sinusoidal positional embeddings E_pos in R^(N x D) are added element-wise to the patch tokens:")

    add_code_block(doc, "E_pos(pos, 2i)   = sin(pos / 10000^(2i / D))\nE_pos(pos, 2i+1) = cos(pos / 10000^(2i / D))\nz = z_0 + E_pos")

    add_h3(doc, "3. Multi-Head Self-Attention (MHSA) Mechanics")
    add_body(doc, "Inside each of the 12 transformer blocks, the token matrix X in R^(N x D) is projected into Query (Q), Key (K), and Value (V) matrices using learned projection weights W_Q, W_K, W_V in R^(D x d_k):")

    add_code_block(doc, "Q = X W_Q,    K = X W_K,    V = X W_V\n\nAttention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V")

    add_body(doc, "Mathematical Role of 1/sqrt(d_k):", bold_prefix="Why Scale by 1/sqrt(d_k)? ")
    add_body(doc, "For large embedding dimensions d_k, the dot products Q K^T grow substantially in magnitude. This pushes the softmax function into regions of extremely small gradients (vanishing gradient problem). Dividing by sqrt(d_k) normalizes the variance of the dot product back to 1.0, preserving healthy gradient flow during backpropagation.")

    add_body(doc, "In ViT-Tiny, the embedding dimension D = 192 is split across h = 3 attention heads, giving each head a feature subspace dimension of d_k = D / h = 192 / 3 = 64 dimensions. The outputs of the 3 heads are concatenated and projected through W_O in R^(D x D):")

    add_code_block(doc, "MHSA(X) = Concat(head_1, head_2, head_3) W_O")

    add_h3(doc, "4. Feed-Forward Network & Pre-LayerNorm")
    add_body(doc, "Each transformer block contains a Feed-Forward Network (MLP) consisting of two linear layers with GELU non-linearity, expanding the dimension by 4x (from 192 to 768, then back to 192). Pre-LayerNorm and residual connections stabilize deep training:")

    add_code_block(doc, "x' = x + MHSA(LayerNorm(x))\nx'' = x' + MLP(LayerNorm(x'))")

    add_h2(doc, "6.2 Masked Autoencoder (MAE) Self-Supervised Learning")
    add_body(doc, "The Masked Autoencoder (He et al., CVPR 2022) adapts self-supervised masked language modeling (like BERT) to computer vision. For chest radiography, MAE solves the core problem of unlabeled data:")

    add_callout(doc, "WHY MAE OUTPERFORMS CONTRASTIVE METHODS ON CHEST X-RAYS",
        "Viva Examination Defense: Why did we choose MAE instead of SimCLR or MoCo?\n\n"
        "1. Contrastive Learning (SimCLR, MoCo) forces two augmented views of an image to have identical representations while pushing other images away. In medical chest radiography, contrastive augmentations like aggressive color jitter, Gaussian blur, or extreme random cropping can destroy subtle diagnostic features (such as faint apical ground-glass opacities or micro-cavitations). Furthermore, pushing different patients apart in embedding space is flawed because many patients share the exact same underlying healthy thoracic pathology.\n"
        "2. Masked Autoencoding treats images as dense continuous signals. By hiding 80% of the image and forcing the network to predict the missing anatomical structures (ribs, cardiac silhouette, diaphragm, lung parenchyma), the encoder is forced to learn real anatomical and spatial correlations without relying on destructive augmentations.",
        "PROFESSOR"
    )

    add_h3(doc, "The Asymmetric Architecture & Masking Mechanics")
    add_bullet(doc, "80% of all patches are randomly masked out. Out of 196 patches, exactly 39 patches are kept visible, and 157 patches are masked.", bold_prefix="1. Heavy Masking Ratio (80%): ")
    add_bullet(doc, "The ViT-Tiny encoder operates EXCLUSIVELY on the 39 visible patches. Because self-attention complexity scales quadratically with sequence length O(N^2), processing 39 tokens instead of 196 yields an enormous (196/39)^2 = 25.2x reduction in attention FLOPs!", bold_prefix="2. Asymmetric ViT Encoder: ")
    add_bullet(doc, "The full token sequence is restored by concatenating the 39 encoded latent tokens with 157 shared, learnable [MASK] tokens. The original 2D spatial arrangement is restored using ids_restore.", bold_prefix="3. Token Restoration: ")
    add_bullet(doc, "A lightweight 4-layer transformer decoder (D_dec = 256, 8 heads) predicts the raw pixel values of all 196 patches.", bold_prefix="4. Lightweight Decoder: ")

    add_h3(doc, "Mathematical Loss Function")
    add_body(doc, "Crucially, the reconstruction Mean Squared Error (MSE) loss is calculated ONLY on the patches that were masked:")

    add_code_block(doc, "L_MAE = (1 / sum(M)) * sum_{i=1}^N M_i * ||pred_i - target_i||^2\n\nwhere M_i in {0, 1} is 1 if patch i was masked, and 0 if patch i was visible.")

    add_body(doc, "If the loss were computed on visible patches as well, the network could minimize loss simply by learning an identity mapping of visible pixels rather than inferring global lung anatomy.")

    add_h2(doc, "6.3 Optimization Mechanics: AdamW, Cosine LR & Gradient Clipping")
    add_body(doc, "Vision transformers lack convolutional inductive biases and are prone to training instability. The optimization stack is calibrated specifically to ensure stable convergence:")

    add_bullet(doc, "Standard Adam applies L2 weight decay to the gradient, which distorts adaptive moment estimates. AdamW (Loshchilov & Hutter, ICLR 2019) decouples weight decay directly into the parameter update step: theta_(t+1) = theta_t - lr * lambda * theta_t - (lr / (sqrt(v_t) + eps)) * m_t.", bold_prefix="1. Decoupled Weight Decay (AdamW): ")
    add_bullet(doc, "Smoothly anneals learning rate from 1e-3 down to 1e-6 following a cosine curve: eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_max)).", bold_prefix="2. Cosine Annealing LR: ")
    add_bullet(doc, "nn.utils.clip_grad_norm_(parameters, max_norm=1.0) rescales parameter gradients if their global L2 norm exceeds 1.0, eliminating gradient spikes caused by sudden large patch reconstruction errors.", bold_prefix="3. Gradient Norm Clipping: ")

    add_horizontal_rule(doc)
