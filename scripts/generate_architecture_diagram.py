"""
scripts/generate_architecture_diagram.py
-----------------------------------------
Generates the publication-grade IEEE System Architecture Diagram for Fed-SSL-TB.
Accurately depicts:
  1. Hospital Clients (N=10) with CXR patches, 80% masking, ViT-Tiny MAE, and DP-SGD.
  2. Central Federated Server with FedAvg aggregation, differential privacy, and global broadcast.
  3. Few-Shot Prototypical Metric Adaptation at edge clinics with K=20 support and peak 84.97% AUC.
"""

import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("paper_assets/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\kiran\.gemini\antigravity-ide\brain\70966951-cea3-465c-869b-a6f9d00b1ac5")

def generate_system_architecture():
    print("[Generating] Fed-SSL-TB High-Resolution System Architecture Diagram...")
    fig = plt.figure(figsize=(16, 11.2), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Load real CXR thumbnail
    cxr_path = Path("data/raw/Shenzhen/images/images/CHNCXR_0001_0.png")
    cxr_img = None
    if cxr_path.exists():
        try:
            cxr_raw = Image.open(cxr_path).convert("L")
            cxr_raw.thumbnail((120, 120))
            cxr_img = np.array(cxr_raw)
        except Exception as e:
            print("CXR thumbnail notice:", e)

    # =========================================================================
    # 1. TOP TIER: STAGE 2 - FEW-SHOT PROTOTYPICAL ADAPTATION (EDGE CLINICS)
    # =========================================================================
    stage2_bg = FancyBboxPatch((2, 70), 96, 28, boxstyle="round,pad=0.6,rounding_size=1.0",
                               facecolor='#f8fafc', edgecolor='#b91c1c', linewidth=2.0)
    ax.add_patch(stage2_bg)

    # Title & Subtitle
    ax.text(4, 95.5, "STAGE 2: Low-Resource Prototypical TB Metric Adaptation (Edge Clinical Deployment)",
            fontsize=12.5, fontweight='bold', color='#991b1b')
    ax.text(4, 93.2, "Zero-Backpropagation Diagnostic Triage with Extreme Label Scarcity (K = 20 Reference Scans per Class)",
            fontsize=9.0, fontstyle='italic', color='#475569')

    # Box 2A: Reference Support Set
    supp_box = FancyBboxPatch((4, 72.5), 27, 18.5, boxstyle="round,pad=0.4",
                              facecolor='#ffffff', edgecolor='#ea580c', linewidth=1.4)
    ax.add_patch(supp_box)
    ax.text(17.5, 88.5, "Reference Support Cohort (K-Shot)", ha='center', fontsize=9.5, fontweight='bold', color='#9a3412')
    ax.text(17.5, 86.3, "Shenzhen Hospital (Confirmed Ground Truth)", ha='center', fontsize=7.8, color='#64748b')

    norm_chip = FancyBboxPatch((5.5, 80), 24, 5.0, boxstyle="round,pad=0.3",
                               facecolor='#f0fdf4', edgecolor='#86efac', linewidth=1.0)
    ax.add_patch(norm_chip)
    ax.text(17.5, 83.2, "Normal Reference: K = 20 Scans", ha='center', fontsize=8.2, fontweight='bold', color='#166534')
    ax.text(17.5, 81.3, "Confirmed healthy chest radiographs", ha='center', fontsize=7.2, color='#15803d')

    tb_chip = FancyBboxPatch((5.5, 73.8), 24, 5.0, boxstyle="round,pad=0.3",
                             facecolor='#fef2f2', edgecolor='#fca5a5', linewidth=1.0)
    ax.add_patch(tb_chip)
    ax.text(17.5, 77.0, "TB Positive Reference: K = 20 Scans", ha='center', fontsize=8.2, fontweight='bold', color='#991b1b')
    ax.text(17.5, 75.1, "Microbiologically confirmed pulmonary TB", ha='center', fontsize=7.2, color='#b91c1c')

    # Arrow 2A -> 2B
    ax.annotate('', xy=(34.5, 81.7), xytext=(31.5, 81.7),
                arrowprops=dict(arrowstyle="-|>", lw=2, color='#ea580c'))

    # Box 2B: Frozen Feature Space & Prototypes
    proto_box = FancyBboxPatch((35, 72.5), 31, 18.5, boxstyle="round,pad=0.4",
                               facecolor='#ffffff', edgecolor='#0284c7', linewidth=1.4)
    ax.add_patch(proto_box)
    ax.text(50.5, 88.5, "Frozen Metric Space & Prototypes", ha='center', fontsize=9.5, fontweight='bold', color='#0369a1')
    ax.text(50.5, 86.3, "Zero Backpropagation / 2-Layer Head (192 -> 128)", ha='center', fontsize=7.8, color='#64748b')

    enc_chip = FancyBboxPatch((36.5, 80), 28, 5.0, boxstyle="round,pad=0.3",
                              facecolor='#f0f9ff', edgecolor='#7dd3fc', linewidth=1.0)
    ax.add_patch(enc_chip)
    ax.text(50.5, 83.2, "Frozen Global Encoder (f_theta*)", ha='center', fontsize=8.2, fontweight='bold', color='#0369a1')
    ax.text(50.5, 81.3, "Extracts 192-dim latent vector: z = f_theta*(x)", ha='center', fontsize=7.2, color='#0284c7')

    proto_chip = FancyBboxPatch((36.5, 73.8), 28, 5.0, boxstyle="round,pad=0.3",
                                facecolor='#e0f2fe', edgecolor='#38bdf8', linewidth=1.0)
    ax.add_patch(proto_chip)
    ax.text(50.5, 77.0, "Class Prototype Centroids (c_0, c_1)", ha='center', fontsize=8.2, fontweight='bold', color='#075985')
    ax.text(50.5, 75.1, "c_c = (1/K) sum z_i (Angular separation: 52 deg)", ha='center', fontsize=7.2, color='#0369a1')

    # Arrow 2B -> 2C
    ax.annotate('', xy=(69.5, 81.7), xytext=(66.5, 81.7),
                arrowprops=dict(arrowstyle="-|>", lw=2, color='#0284c7'))

    # Box 2C: Clinical Query Triage & Benchmarks
    eval_box = FancyBboxPatch((70, 72.5), 26, 18.5, boxstyle="round,pad=0.4",
                              facecolor='#ffffff', edgecolor='#059669', linewidth=1.4)
    ax.add_patch(eval_box)
    ax.text(83, 88.5, "Query Classification & Triage", ha='center', fontsize=9.5, fontweight='bold', color='#065f46')
    ax.text(83, 86.3, "Scaled Cosine Metric Softmax (tau = 10.0)", ha='center', fontsize=7.8, color='#64748b')

    bench_chip = FancyBboxPatch((71.5, 80), 23, 5.0, boxstyle="round,pad=0.3",
                                facecolor='#ecfdf5', edgecolor='#6ee7b7', linewidth=1.0)
    ax.add_patch(bench_chip)
    ax.text(83, 83.2, "Peak Shenzhen In-Domain Benchmark", ha='center', fontsize=8.0, fontweight='bold', color='#065f46')
    ax.text(83, 81.3, "ROC-AUC: 84.97% | Acc: 80.29% | Spec: 84.39%", ha='center', fontsize=7.0, color='#047857')

    stress_chip = FancyBboxPatch((71.5, 73.8), 23, 5.0, boxstyle="round,pad=0.3",
                                 facecolor='#fef3c7', edgecolor='#fcd34d', linewidth=1.0)
    ax.add_patch(stress_chip)
    ax.text(83, 77.0, "Montgomery Cross-Site Stress Test", ha='center', fontsize=8.0, fontweight='bold', color='#92400e')
    ax.text(83, 75.1, "Preserves 60.34% recall (ImageNet drops to 27%)", ha='center', fontsize=7.0, color='#b45309')

    # =========================================================================
    # 2. MIDDLE TIER: CENTRAL FEDERATED SERVER
    # =========================================================================
    # Transfer Banner between Stage 2 and Server
    ax.annotate('', xy=(50, 70), xytext=(50, 65.5),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color='#b91c1c'))
    ax.text(50, 67.8, "Transfer Pretrained Global Encoder Weights theta* to Edge Clinics (Zero Backpropagation)",
            ha='center', va='center', fontsize=8.4, fontweight='bold', color='#991b1b',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#fef2f2', edgecolor='#fca5a5', lw=1.0))

    server_bg = FancyBboxPatch((22, 45), 56, 19.5, boxstyle="round,pad=0.6,rounding_size=1.0",
                               facecolor='#f5f3ff', edgecolor='#6d28d9', linewidth=2.0)
    ax.add_patch(server_bg)

    ax.text(50, 62.0, "CENTRAL FEDERATED SERVER (Decentralized Aggregation)",
            ha='center', fontsize=11.5, fontweight='bold', color='#5b21b6')
    ax.text(50, 59.8, "Decentralized Coordination across 10 Hospital Silos (100 Federated Rounds Completed)",
            ha='center', fontsize=8.4, color='#6b21a8')

    # Server inner modules
    s1_box = FancyBboxPatch((24, 46.5), 25, 11.5, boxstyle="round,pad=0.3",
                            facecolor='#ffffff', edgecolor='#c4b5fd', linewidth=1.2)
    ax.add_patch(s1_box)
    ax.text(36.5, 55.5, "Parameter Aggregation (FedAvg)", ha='center', fontsize=8.8, fontweight='bold', color='#4c1d95')
    ax.text(36.5, 53.0, "w^(t+1) = sum_(k=1)^10 (n_k / N) w_k^(t+1)", ha='center', fontsize=7.6, color='#5b21b6')
    ax.text(36.5, 50.8, "Client-weighted parameter averaging", ha='center', fontsize=7.2, color='#64748b')
    ax.text(36.5, 48.6, "Convergence: loss 0.658 -> 0.1349", ha='center', fontsize=7.2, color='#059669', fontweight='bold')

    s2_box = FancyBboxPatch((51, 46.5), 25, 11.5, boxstyle="round,pad=0.3",
                            facecolor='#ffffff', edgecolor='#c4b5fd', linewidth=1.2)
    ax.add_patch(s2_box)
    ax.text(63.5, 55.5, "Differential Privacy & Security", ha='center', fontsize=8.8, fontweight='bold', color='#4c1d95')
    ax.text(63.5, 53.0, "Zero Raw Radiograph Sharing", ha='center', fontsize=7.6, color='#b91c1c', fontweight='bold')
    ax.text(63.5, 50.8, "DP-SGD: Clipping C=1.0, Noise sigma=1.1", ha='center', fontsize=7.2, color='#5b21b6')
    ax.text(63.5, 48.6, "HIPAA & GDPR Compliant Security", ha='center', fontsize=7.2, color='#64748b')

    # =========================================================================
    # 3. INTER-TIER ARROWS (HOSPITALS <-> SERVER)
    # =========================================================================
    # Hospital 1 to Server
    ax.annotate('', xy=(27, 45), xytext=(15, 36.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#7c3aed'))
    ax.text(18.5, 42.0, "Local Updates", fontsize=7.5, fontweight='bold', color='#7c3aed',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#ffffff', edgecolor='none'))

    ax.annotate('', xy=(17.5, 36.5), xytext=(29.5, 45),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#059669'))
    ax.text(26.0, 39.5, "Global Parameters", fontsize=7.5, fontweight='bold', color='#059669',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#ffffff', edgecolor='none'))

    # Hospital 2 to Server
    ax.annotate('', xy=(47.5, 45), xytext=(47.5, 36.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#7c3aed'))
    ax.text(44.0, 40.8, "Local Updates", fontsize=7.5, fontweight='bold', color='#7c3aed', rotation=90,
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#ffffff', edgecolor='none'))

    ax.annotate('', xy=(52.5, 36.5), xytext=(52.5, 45),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#059669'))
    ax.text(56.0, 40.8, "Global Parameters", fontsize=7.5, fontweight='bold', color='#059669', rotation=90,
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#ffffff', edgecolor='none'))

    # Hospital 10 to Server
    ax.annotate('', xy=(73, 45), xytext=(85, 36.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#7c3aed'))
    ax.text(81.5, 42.0, "Local Updates", fontsize=7.5, fontweight='bold', color='#7c3aed',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#ffffff', edgecolor='none'))

    ax.annotate('', xy=(82.5, 36.5), xytext=(70.5, 45),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#059669'))
    ax.text(74.0, 39.5, "Global Parameters", fontsize=7.5, fontweight='bold', color='#059669',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#ffffff', edgecolor='none'))

    # =========================================================================
    # 4. BOTTOM TIER: 10 DECENTRALIZED HOSPITAL CLIENT SILOS (STAGE 1)
    # =========================================================================
    stage1_bg = FancyBboxPatch((2, 1.5), 96, 35.0, boxstyle="round,pad=0.6,rounding_size=1.0",
                               facecolor='#f8fafc', edgecolor='#0369a1', linewidth=2.0)
    ax.add_patch(stage1_bg)

    ax.text(4, 34.2, "STAGE 1: Decentralized Federated Self-Supervised Pre-Training (MAE on ViT-Tiny across Unlabeled Radiographs)",
            fontsize=11.5, fontweight='bold', color='#0369a1')
    ax.text(4, 32.2, "Decentralized Silos with Non-IID Dirichlet Skew (No Raw Patient Radiographs Leave Hospital Firewalls)",
            fontsize=8.5, fontstyle='italic', color='#64748b')

    # Hospital Silo 1
    h1_box = FancyBboxPatch((4, 3.5), 28, 27.0, boxstyle="round,pad=0.4",
                            facecolor='#ffffff', edgecolor='#0284c7', linewidth=1.3)
    ax.add_patch(h1_box)
    ax.text(18, 28.0, "Hospital Client 1 (NIH Silo)", ha='center', fontsize=9.2, fontweight='bold', color='#0369a1')
    ax.text(18, 26.2, "Unlabeled Radiographs (n = 12,286)", ha='center', fontsize=7.6, color='#64748b')

    if cxr_img is not None:
        imagebox = OffsetImage(cxr_img, zoom=0.34, cmap='gray')
        ab = AnnotationBbox(imagebox, (8.5, 19.8), frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor='#cbd5e1', lw=0.8))
        ax.add_artist(ab)
        ax.text(8.5, 14.5, "Raw X-Ray\n(Local)", ha='center', fontsize=6.8, color='#475569')

    mae1_box = FancyBboxPatch((13.8, 14.8), 16.8, 10.2, boxstyle="round,pad=0.25",
                              facecolor='#f0fdf4', edgecolor='#86efac', linewidth=1.0)
    ax.add_patch(mae1_box)
    ax.text(22.2, 23.0, "MAE Pre-Training", ha='center', fontsize=7.8, fontweight='bold', color='#166534')
    ax.text(22.2, 21.2, "• 80% Patch Mask (16x16)", ha='center', fontsize=6.8, color='#15803d')
    ax.text(22.2, 19.6, "• ViT-Tiny Encoder (192-d)", ha='center', fontsize=6.8, color='#15803d')
    ax.text(22.2, 18.0, "• Decoder (4 blk, 128-d)", ha='center', fontsize=6.8, color='#15803d')
    ax.text(22.2, 16.4, "• Reconstruction Loss", ha='center', fontsize=6.8, color='#15803d')

    dp1_box = FancyBboxPatch((5.5, 5.0), 25, 7.8, boxstyle="round,pad=0.25",
                             facecolor='#f5f3ff', edgecolor='#c4b5fd', linewidth=1.0)
    ax.add_patch(dp1_box)
    ax.text(18, 10.8, "Differential Privacy (DP-SGD)", ha='center', fontsize=7.6, fontweight='bold', color='#5b21b6')
    ax.text(18, 9.0, "Clip C=1.0, Noise sigma=1.1, RDP State", ha='center', fontsize=6.8, color='#6b21a8')
    ax.text(18, 7.3, "Strict Privacy Boundary", ha='center', fontsize=6.8, color='#15803d', fontweight='bold')

    # Hospital Silo 2
    h2_box = FancyBboxPatch((36, 3.5), 28, 27.0, boxstyle="round,pad=0.4",
                            facecolor='#ffffff', edgecolor='#0284c7', linewidth=1.3)
    ax.add_patch(h2_box)
    ax.text(50, 28.0, "Hospital Client 2 (NIH Silo)", ha='center', fontsize=9.2, fontweight='bold', color='#0369a1')
    ax.text(50, 26.2, "Unlabeled Radiographs (n = 7,738)", ha='center', fontsize=7.6, color='#64748b')

    if cxr_img is not None:
        imagebox2 = OffsetImage(cxr_img, zoom=0.34, cmap='gray')
        ab2 = AnnotationBbox(imagebox2, (40.5, 19.8), frameon=True, pad=0.1,
                             bboxprops=dict(edgecolor='#cbd5e1', lw=0.8))
        ax.add_artist(ab2)
        ax.text(40.5, 14.5, "Raw X-Ray\n(Local)", ha='center', fontsize=6.8, color='#475569')

    mae2_box = FancyBboxPatch((45.8, 14.8), 16.8, 10.2, boxstyle="round,pad=0.25",
                              facecolor='#f0fdf4', edgecolor='#86efac', linewidth=1.0)
    ax.add_patch(mae2_box)
    ax.text(54.2, 23.0, "MAE Pre-Training", ha='center', fontsize=7.8, fontweight='bold', color='#166534')
    ax.text(54.2, 21.2, "• 80% Patch Mask (16x16)", ha='center', fontsize=6.8, color='#15803d')
    ax.text(54.2, 19.6, "• ViT-Tiny Encoder (192-d)", ha='center', fontsize=6.8, color='#15803d')
    ax.text(54.2, 18.0, "• Decoder (4 blk, 128-d)", ha='center', fontsize=6.8, color='#15803d')
    ax.text(54.2, 16.4, "• Reconstruction Loss", ha='center', fontsize=6.8, color='#15803d')

    dp2_box = FancyBboxPatch((37.5, 5.0), 25, 7.8, boxstyle="round,pad=0.25",
                             facecolor='#f5f3ff', edgecolor='#c4b5fd', linewidth=1.0)
    ax.add_patch(dp2_box)
    ax.text(50, 10.8, "Differential Privacy (DP-SGD)", ha='center', fontsize=7.6, fontweight='bold', color='#5b21b6')
    ax.text(50, 9.0, "Clip C=1.0, Noise sigma=1.1, RDP State", ha='center', fontsize=6.8, color='#6b21a8')
    ax.text(50, 7.3, "Strict Privacy Boundary", ha='center', fontsize=6.8, color='#15803d', fontweight='bold')

    # Hospital Silo 10
    h10_box = FancyBboxPatch((68, 3.5), 28, 27.0, boxstyle="round,pad=0.4",
                             facecolor='#ffffff', edgecolor='#0284c7', linewidth=1.3)
    ax.add_patch(h10_box)
    ax.text(82, 28.0, "Hospital Client 10 (NIH Silo)", ha='center', fontsize=9.2, fontweight='bold', color='#0369a1')
    ax.text(82, 26.2, "Non-IID Dirichlet Skew: alpha = 1.0 (n = 308)", ha='center', fontsize=7.6, color='#64748b')

    if cxr_img is not None:
        imagebox10 = OffsetImage(cxr_img, zoom=0.34, cmap='gray')
        ab10 = AnnotationBbox(imagebox10, (72.5, 19.8), frameon=True, pad=0.1,
                              bboxprops=dict(edgecolor='#cbd5e1', lw=0.8))
        ax.add_artist(ab10)
        ax.text(72.5, 14.5, "Raw X-Ray\n(Local)", ha='center', fontsize=6.8, color='#475569')

    mae10_box = FancyBboxPatch((77.8, 14.8), 16.8, 10.2, boxstyle="round,pad=0.25",
                               facecolor='#f0fdf4', edgecolor='#86efac', linewidth=1.0)
    ax.add_patch(mae10_box)
    ax.text(86.2, 23.0, "MAE Pre-Training", ha='center', fontsize=7.8, fontweight='bold', color='#166534')
    ax.text(86.2, 21.2, "• 80% Patch Mask (16x16)", ha='center', fontsize=6.8, color='#15803d')
    ax.text(86.2, 19.6, "• ViT-Tiny Encoder (192-d)", ha='center', fontsize=6.8, color='#15803d')
    ax.text(86.2, 18.0, "• Decoder (4 blk, 128-d)", ha='center', fontsize=6.8, color='#15803d')
    ax.text(86.2, 16.4, "• Reconstruction Loss", ha='center', fontsize=6.8, color='#15803d')

    dp10_box = FancyBboxPatch((69.5, 5.0), 25, 7.8, boxstyle="round,pad=0.25",
                              facecolor='#f5f3ff', edgecolor='#c4b5fd', linewidth=1.0)
    ax.add_patch(dp10_box)
    ax.text(82, 10.8, "Differential Privacy (DP-SGD)", ha='center', fontsize=7.6, fontweight='bold', color='#5b21b6')
    ax.text(82, 9.0, "Clip C=1.0, Noise sigma=1.1, RDP State", ha='center', fontsize=6.8, color='#6b21a8')
    ax.text(82, 7.3, "Strict Privacy Boundary", ha='center', fontsize=6.8, color='#15803d', fontweight='bold')

    # Ellipsis dots between Hospital 2 and 10
    ax.text(66.0, 17.0, "• • •", ha='center', va='center', fontsize=18, fontweight='bold', color='#94a3b8')

    # Save high-resolution outputs
    plt.tight_layout()
    png_path = OUTPUT_DIR / "fedssl_tb_system_architecture.png"
    pdf_path = OUTPUT_DIR / "fedssl_tb_system_architecture.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    # Copy to artifact directory for UI display
    try:
        shutil.copy(png_path, ARTIFACT_DIR / "fedssl_tb_system_architecture.png")
    except Exception as e:
        print("Artifact copy notice:", e)

    print(f"  [SUCCESS] Saved 300 DPI architecture diagram to:\n    - {png_path}\n    - {pdf_path}")

if __name__ == "__main__":
    generate_system_architecture()
