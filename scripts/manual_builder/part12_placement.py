"""
scripts/manual_builder/part12_placement.py
------------------------------------------
Part XII: Placement Mastery Through This Project — Technical Interview Challenges,
Read-Code Exercises, Predict-the-Output, and Live Coding Tasks with Solutions.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part12(doc):
    add_h1(doc, "Part XII: Technical Placement & Interview Mastery")

    add_h2(doc, "12.1 The ML Engineer & Systems Interview Framework")
    add_body(doc, "In top software engineering and AI placement interviews (Google, Microsoft, Amazon, Healthcare AI labs), interviewers will probe this project to evaluate three distinct engineering competencies:")
    add_bullet(doc, "Can you trace tensor dimensions, autograd computation graphs, and custom loss functions?", bold_prefix="1. PyTorch & Tensor Fluency: ")
    add_bullet(doc, "Can you design distributed client-server systems, handle non-IID heterogeneity, and reason about network bottlenecks?", bold_prefix="2. Distributed & FL Architecture: ")
    add_bullet(doc, "Do you understand diagnostic sensitivity, class imbalance, domain shift, and medical regulations?", bold_prefix="3. Clinical ML Rigor: ")

    add_h2(doc, "12.2 Read-Code & Predict-the-Output Challenges")

    add_h3(doc, "Challenge 1: Spot the Autograd Bug in Federated Optimization")
    add_body(doc, "Question: An engineer writes the following FedProx proximal loss calculation in PyTorch. Will the model parameters receive proximal gradients? Why or why not?")

    code_bug1 = (
        "# Defective code snippet:\n"
        "proximal = 0.0\n"
        "for name, param in model.named_parameters():\n"
        "    ref = global_weights[name].detach()\n"
        "    proximal += torch.sum((param.data - ref) ** 2)\n"
        "loss = mae_loss + (mu / 2.0) * proximal\n"
        "loss.backward()"
    )
    add_code_block(doc, code_bug1)

    add_body(doc, "Model Answer:", bold_prefix="Answer: ")
    add_body(doc, "NO! The parameters will receive ZERO gradients from the proximal loss. Accessing param.data bypasses PyTorch's autograd tracking engine completely. The tensor operation (param.data - ref) produces a tensor with requires_grad=False. During loss.backward(), gradients will flow only through mae_loss, completely disabling the FedProx penalty! The correct implementation must operate directly on the live parameter tensor: (param - ref) ** 2.")

    add_h3(doc, "Challenge 2: Predict the Output Tensor Shapes")
    add_body(doc, "Question: Given an image tensor x with shape (8, 3, 224, 224), patch_size = 16, embed_dim = 192, and mask_ratio = 0.75, what are the exact output shapes of:")
    add_bullet(doc, "target = patchify(x)", bold_prefix="a) ")
    add_bullet(doc, "visible_tokens, mask, ids_restore = random_masking(tokens, 0.75)", bold_prefix="b) ")
    add_bullet(doc, "encoded = _encode_visible(visible_tokens)", bold_prefix="c) ")
    add_bullet(doc, "pred = decoder(encoded, ids_restore)", bold_prefix="d) ")

    add_body(doc, "Model Answer:", bold_prefix="Answer: ")
    add_bullet(doc, "target.shape = (8, 196, 768) — 196 patches of 16*16*3 = 768 pixels.", bold_prefix="a) ")
    add_bullet(doc, "visible_tokens.shape = (8, 49, 192) — 25% of 196 = 49 visible tokens. mask.shape = (8, 196). ids_restore.shape = (8, 196).", bold_prefix="b) ")
    add_bullet(doc, "encoded.shape = (8, 49, 192) — ViT-Tiny encoder processes only the 49 visible tokens.", bold_prefix="c) ")
    add_bullet(doc, "pred.shape = (8, 196, 768) — Decoder inserts 147 mask tokens, restores order, and reconstructs all 196 patches.", bold_prefix="d) ")

    add_h2(doc, "12.3 Three Practical Live Coding Tasks (With Worked Solutions)")

    add_h3(doc, "Task 1: Implement Patchify and Unpatchify from Scratch in PyTorch")
    add_body(doc, "Problem: Write pure PyTorch functions to partition a batch of images (B, C, H, W) into non-overlapping patches (B, N, P*P*C) and reconstruct the image back from patch predictions.")

    code_sol1 = (
        "def patchify(x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:\n"
        "    B, C, H, W = x.shape\n"
        "    assert H == W and H % patch_size == 0\n"
        "    h = w = H // patch_size\n"
        "    # (B, C, h, p, w, p) -> (B, h, w, p, p, C) -> (B, h*w, p*p*C)\n"
        "    x = x.reshape(B, C, h, patch_size, w, patch_size)\n"
        "    x = x.permute(0, 2, 4, 3, 5, 1)\n"
        "    patches = x.reshape(B, h * w, patch_size * patch_size * C)\n"
        "    return patches\n"
        "\n"
        "def unpatchify(patches: torch.Tensor, patch_size: int = 16, in_channels: int = 3) -> torch.Tensor:\n"
        "    B, N, _ = patches.shape\n"
        "    h = w = int(N ** 0.5)\n"
        "    # (B, h, w, p, p, C) -> (B, C, h, p, w, p) -> (B, C, h*p, w*p)\n"
        "    x = patches.reshape(B, h, w, patch_size, patch_size, in_channels)\n"
        "    x = x.permute(0, 5, 1, 3, 2, 4)\n"
        "    img = x.reshape(B, in_channels, h * patch_size, w * patch_size)\n"
        "    return img"
    )
    add_code_block(doc, code_sol1)

    add_h3(doc, "Task 2: Implement Prototypical Loss with Temperature Cosine Metric")
    add_body(doc, "Problem: Given query embeddings (N, D), query labels (N), and class prototypes (C, D), implement prototypical cross-entropy loss with cosine similarity scaled by temperature tau = 10.0.")

    code_sol2 = (
        "import torch.nn.functional as F\n\n"
        "def prototypical_cosine_loss(queries: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, tau: float = 10.0):\n"
        "    # L2 normalize along embedding dimension\n"
        "    queries_norm = F.normalize(queries, p=2, dim=-1)      # (N, D)\n"
        "    prototypes_norm = F.normalize(prototypes, p=2, dim=-1)  # (C, D)\n"
        "    # Compute cosine similarity matrix scaled by tau\n"
        "    logits = torch.mm(queries_norm, prototypes_norm.t()) * tau  # (N, C)\n"
        "    # Standard Cross-Entropy over distance logits\n"
        "    loss = F.cross_entropy(logits, labels.long())\n"
        "    probs = F.softmax(logits, dim=-1)\n"
        "    return loss, probs"
    )
    add_code_block(doc, code_sol2)

    add_horizontal_rule(doc)
