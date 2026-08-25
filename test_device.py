"""
test_device.py
--------------
Device consistency tests for the PrototypicalHead and few-shot training pipeline.
Tests CPU, CUDA, and mixed-device scenarios.
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.proto_head import PrototypicalHead

EMBED_DIM = 192
NUM_CLASSES = 2
PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        FAIL += 1


# ─── Test Suite ──────────────────────────────────────────────────────────────

def run_proto_test_on_device(device_name, device):
    """Test prototypical head entirely on one device."""
    ph = PrototypicalHead(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES).to(device)
    supp_emb = torch.randn(10, EMBED_DIM, device=device)
    supp_lbl = torch.tensor([0]*5 + [1]*5, device=device)
    query_emb = torch.randn(6, EMBED_DIM, device=device, requires_grad=True)
    query_lbl = torch.tensor([0, 1, 0, 1, 0, 1], device=device)

    # compute_prototypes
    protos = ph.compute_prototypes(supp_emb, supp_lbl)
    assert protos.device == device, f"Prototypes on {protos.device}, expected {device}"
    assert protos.shape == (NUM_CLASSES, EMBED_DIM)

    # forward (with explicit prototypes)
    probs = ph.forward(query_emb, protos)
    assert probs.device == device
    assert probs.shape == (6, NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(6, device=device), atol=1e-5)

    # predict
    labels, probs2 = ph.predict(query_emb, protos)
    assert labels.device == device
    assert probs2.device == device

    # prototypical_loss
    loss, probs3 = ph.prototypical_loss(query_emb, query_lbl, protos)
    assert loss.device == device
    assert loss.requires_grad  # loss should have grad from the log+nll computation

    # get_learnable_prototypes
    learnable_protos = ph.get_learnable_prototypes(supp_emb, supp_lbl)
    assert learnable_protos.device == device


def test_cpu():
    run_proto_test_on_device("CPU", torch.device("cpu"))


def test_cuda():
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return
    run_proto_test_on_device("CUDA", torch.device("cuda"))


def test_mixed_queries_cuda_protos_cpu():
    """Queries on CUDA, prototypes on CPU — should auto-resolve."""
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available for mixed test")
        return
    ph = PrototypicalHead(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES)
    # Prototypes computed on CPU
    supp_emb = torch.randn(10, EMBED_DIM)
    supp_lbl = torch.tensor([0]*5 + [1]*5)
    protos = ph.compute_prototypes(supp_emb, supp_lbl)
    assert protos.device == torch.device("cpu")

    # Queries on CUDA
    query_emb = torch.randn(6, EMBED_DIM, device="cuda")
    query_lbl = torch.tensor([0, 1, 0, 1, 0, 1], device="cuda")

    # This should NOT crash — protos should be moved to CUDA internally
    probs = ph.forward(query_emb, protos)
    assert probs.device.type == "cuda"

    # Loss should also work
    loss, _ = ph.prototypical_loss(query_emb, query_lbl, protos)
    assert loss.device.type == "cuda"


def test_mixed_queries_cpu_protos_cuda():
    """Queries on CPU, prototypes on CUDA — should auto-resolve."""
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available for mixed test")
        return
    ph = PrototypicalHead(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES)
    supp_emb = torch.randn(10, EMBED_DIM, device="cuda")
    supp_lbl = torch.tensor([0]*5 + [1]*5, device="cuda")
    protos = ph.compute_prototypes(supp_emb, supp_lbl)

    query_emb = torch.randn(6, EMBED_DIM)  # CPU
    query_lbl = torch.tensor([0, 1, 0, 1, 0, 1])

    probs = ph.forward(query_emb, protos)
    assert probs.device == torch.device("cpu")  # follows queries

    loss, _ = ph.prototypical_loss(query_emb, query_lbl, protos)
    assert loss.device == torch.device("cpu")


def test_stored_prototypes_device():
    """When using stored prototypes, ensure they're moved to query device."""
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return
    ph = PrototypicalHead(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES)
    # Compute prototypes on CPU (stored internally)
    supp_emb = torch.randn(10, EMBED_DIM)
    supp_lbl = torch.tensor([0]*5 + [1]*5)
    ph.compute_prototypes(supp_emb, supp_lbl)

    # Query on CUDA, using stored prototypes (None)
    query_emb = torch.randn(6, EMBED_DIM, device="cuda")
    probs = ph.forward(query_emb)  # should use stored prototypes, auto-move to CUDA
    assert probs.device.type == "cuda"


def test_gradient_flow_through_encoder():
    """Verify that prototypical loss backpropagates through encoder."""
    from src.models.encoder import get_encoder
    encoder = get_encoder(backbone="vit_tiny", embed_dim=EMBED_DIM)
    encoder.train()

    ph = PrototypicalHead(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES)

    # Create tiny "images"
    support_imgs = torch.randn(4, 3, 224, 224)
    support_labels = torch.tensor([0, 0, 1, 1])
    query_imgs = torch.randn(4, 3, 224, 224)
    query_labels = torch.tensor([0, 1, 0, 1])

    # Forward through encoder (with grad)
    support_emb = encoder(support_imgs)
    protos = ph.get_learnable_prototypes(support_emb, support_labels)
    query_emb = encoder(query_imgs)
    loss, probs = ph.prototypical_loss(query_emb, query_labels, protos)

    # Verify loss has gradient
    assert loss.requires_grad, "Loss should require grad"
    assert loss.grad_fn is not None, "Loss should have grad_fn"

    loss.backward()

    # Verify encoder parameters received gradients
    grads_found = 0
    total_params = 0
    for p in encoder.parameters():
        total_params += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            grads_found += 1

    assert grads_found > 0, f"No encoder params received gradients ({total_params} params checked)"


def test_prototypical_loss_values_change():
    """Verify that prototypical loss changes after an optimizer step."""
    from src.models.encoder import get_encoder
    from torch.optim import AdamW

    encoder = get_encoder(backbone="vit_tiny", embed_dim=EMBED_DIM)
    encoder.train()
    ph = PrototypicalHead(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES)
    optimizer = AdamW(encoder.parameters(), lr=1e-3)

    imgs = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    # Step 1
    optimizer.zero_grad()
    emb = encoder(imgs)
    protos = ph.get_learnable_prototypes(emb[:4], labels[:4])
    loss1, _ = ph.prototypical_loss(emb[4:], labels[4:], protos)
    loss1_val = loss1.item()
    loss1.backward()
    optimizer.step()

    # Step 2
    optimizer.zero_grad()
    emb = encoder(imgs)
    protos = ph.get_learnable_prototypes(emb[:4], labels[:4])
    loss2, _ = ph.prototypical_loss(emb[4:], labels[4:], protos)
    loss2_val = loss2.item()

    assert loss1_val != loss2_val, (
        f"Loss did not change after optimizer step! "
        f"Step 1: {loss1_val:.6f}, Step 2: {loss2_val:.6f}"
    )


def test_dry_run_pipeline():
    """End-to-end dry-run smoke test of the simulation pipeline."""
    import copy
    from src.utils.config import load_config
    from src.server.server import FederatedServer
    from src.client.ssl_train import ssl_local_train
    from src.client.local_train import finetune_local, evaluate_on_montgomery

    # Build minimal config
    config = load_config("configs/kaggle.yaml")
    config.ssl.epochs_per_round = 1
    config.ssl.batch_size = 4
    config.ssl.lr = 1e-3
    config.federated.rounds = 1
    config.finetuning.epochs = 2
    config.finetuning.few_shot_k = 2
    config.finetuning.lr = 1e-3

    device = torch.device("cpu")

    # Server
    server = FederatedServer(config, device=device)
    global_model = server.initialize_global_model()
    global_weights = server.broadcast()

    # Synthetic NIH dataset
    class TinyNIH(torch.utils.data.Dataset):
        def __len__(self): return 16
        def __getitem__(self, i): return torch.randn(3, 224, 224)

    loader = torch.utils.data.DataLoader(TinyNIH(), batch_size=4)
    result = ssl_local_train(
        hospital_id=1,
        model=copy.deepcopy(global_model),
        dataloader=loader,
        config=config,
        global_weights=global_weights,
        device=device,
    )
    assert "encoder_weights" in result
    assert result["num_samples"] > 0

    # Aggregate
    agg = server.aggregate([result["encoder_weights"]], [result["num_samples"]])
    server.update_global_model(agg)

    # Synthetic labeled dataset
    class TinyLabeled(torch.utils.data.Dataset):
        def __init__(self, size=20):
            self.size = size
            self.labels = [i % 2 for i in range(size)]
        def __len__(self): return self.size
        def __getitem__(self, i):
            return torch.randn(3, 224, 224), self.labels[i]
        def get_labels(self): return self.labels

    shenzhen_loader = torch.utils.data.DataLoader(TinyLabeled(20), batch_size=4)
    montgomery_loader = torch.utils.data.DataLoader(TinyLabeled(10), batch_size=4)

    # Fine-tune
    encoder_copy = copy.deepcopy(server.get_encoder()).to(device)
    proto_head, finetune_metrics = finetune_local(
        hospital_id=0,
        encoder=encoder_copy,
        shenzhen_loader=shenzhen_loader,
        config=config,
        device=device,
    )
    assert "auc" in finetune_metrics, "Fine-tuning should produce metrics"

    # Evaluate on Montgomery
    eval_metrics = evaluate_on_montgomery(
        encoder=encoder_copy,
        proto_head=proto_head,
        montgomery_loader=montgomery_loader,
        support_loader=shenzhen_loader,
        config=config,
        device=device,
    )
    assert "auc" in eval_metrics, "Evaluation should produce metrics"


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FedSSL Device & Gradient Tests")
    print("=" * 60)

    print("\n[1] Prototypical Head — CPU")
    test("CPU basic operations", test_cpu)

    print("\n[2] Prototypical Head — CUDA")
    test("CUDA basic operations", test_cuda)

    print("\n[3] Mixed Device — Queries CUDA, Protos CPU")
    test("Mixed device (Q=CUDA, P=CPU)", test_mixed_queries_cuda_protos_cpu)

    print("\n[4] Mixed Device — Queries CPU, Protos CUDA")
    test("Mixed device (Q=CPU, P=CUDA)", test_mixed_queries_cpu_protos_cuda)

    print("\n[5] Stored Prototypes — Cross-device")
    test("Stored prototypes cross-device", test_stored_prototypes_device)

    print("\n[6] Gradient Flow Through Encoder")
    test("Encoder gradient flow", test_gradient_flow_through_encoder)

    print("\n[7] Loss Changes After Optimizer Step")
    test("Loss changes after step", test_prototypical_loss_values_change)

    print("\n[8] End-to-End Dry Run Pipeline")
    test("Dry-run pipeline", test_dry_run_pipeline)

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    if FAIL > 0:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")
