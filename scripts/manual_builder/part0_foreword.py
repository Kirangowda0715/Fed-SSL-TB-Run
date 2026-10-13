"""
scripts/manual_builder/part0_foreword.py
----------------------------------------
Title page, Executive Summary, Evidence Boundary, and Forensic Reconciliation
against the existing 32-page study notes.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule,
    FONT_BODY, FONT_HEADING, COLOR_NAVY_DARK, COLOR_NAVY_MID
)
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def build_part0(doc):
    # ── Title Page ───────────────────────────────────────────────────────────
    for _ in range(2):
        doc.add_paragraph().paragraph_format.space_after = Pt(0)

    p_badge = doc.add_paragraph()
    p_badge.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_badge = p_badge.add_run("FEDERATED SELF-SUPERVISED LEARNING FOR TUBERCULOSIS DETECTION")
    r_badge.font.name = FONT_HEADING
    r_badge.font.size = Pt(13)
    r_badge.font.color.rgb = COLOR_NAVY_MID
    r_badge.bold = True
    p_badge.paragraph_format.space_after = Pt(12)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("FedSSL-TB Engineering Mastery Manual\n& Scientific Defense Guide")
    r_title.font.name = FONT_HEADING
    r_title.font.size = Pt(26)
    r_title.font.color.rgb = COLOR_NAVY_DARK
    r_title.bold = True
    p_title.paragraph_format.space_after = Pt(16)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("A Code-Verified, Rigorous Technical Treatise on Decentralized Pretraining,\nFew-Shot Metric Adaptation, and Cross-Dataset Clinical Diagnostic Reality")
    r_sub.font.name = FONT_BODY
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    p_sub.paragraph_format.space_after = Pt(28)

    # Repository & Environment Metadata Table
    headers = ["Audit Parameter", "Verified System State", "Evidence Anchor"]
    rows = [
        ["Repository Worktree", "d:\\Projects\\1.Final Year Project\\Federated-SSL", "VERIFIED IN SOURCE: Local Workspace"],
        ["Active Git Branch", "dev (tracking second/dev)", "VERIFIED IN EXECUTION: git branch -a -v"],
        ["Verified Commit SHA", "a6354aa383c1ae43b008eb20ab6030cc456ba063", "VERIFIED IN EXECUTION: git rev-parse HEAD"],
        ["Primary Remote (Upstream)", "https://github.com/nmit-1NT23CS111/Fed-SSL-TB.git", "VERIFIED IN SOURCE: git remote -v (origin)"],
        ["Personal Run Remote", "https://github.com/Kirangowda0715/Fed-SSL-TB-Run.git", "VERIFIED IN SOURCE: git remote -v (second)"],
        ["Primary Model Architecture", "ViT-Tiny (patch 16x16, D=192) + MAE Decoder (D=256)", "VERIFIED IN SOURCE: src/models/mae.py"],
        ["Few-Shot Classifier", "Trainable Projection Head + Cosine Sim (tau=10.0)", "VERIFIED IN SOURCE: src/models/proto_head.py"],
        ["Aggregation Protocol", "FedProx (mu=0.01) on Stage 1 Encoder + Decoder", "VERIFIED IN SOURCE: src/server/aggregator.py"],
        ["Hardware & Compute Targets", "Local NVIDIA CUDA GPU + Kaggle Cloud Accelerator", "OBSERVED EXPERIMENT: 100-round GPU run"],
    ]
    add_table(doc, headers, rows, [1.8, 2.7, 2.0])

    doc.add_page_break()

    # ── Executive Summary & Evidence Boundary ────────────────────────────────
    add_h1(doc, "Executive Summary & Scientific Evidence Boundary")
    add_body(doc, "This engineering manual is a comprehensive, code-verified technical authority on the Federated Self-Supervised Learning for Tuberculosis Detection (FedSSL-TB) project. It bridges the gap between high-level architectural concepts and line-by-line runtime execution. Designed for deep engineering mastery, technical placement interviews, research viva defense, and IEEE paper publication, every mathematical formula, tensor transformation, configuration key, and empirical finding in this guide is anchored directly to the verified source code of the repository.")

    add_callout(doc, "STANDARDIZED SCIENTIFIC EVIDENCE TAXONOMY",
        "To ensure uncompromising research integrity, every assertion across this document is tagged with one of the following strict evidence labels:\n"
        "• VERIFIED IN SOURCE: Directly established by inspecting the active Python implementation in the dev branch.\n"
        "• VERIFIED IN EXECUTION: Confirmed by executing live tests, commands, or forward/backward passes.\n"
        "• OBSERVED EXPERIMENT: Supported by concrete saved JSON logs, metrics files, or model checkpoints in the repository.\n"
        "• REPORTED IN EXISTING NOTES: Extracted from earlier study notes but requiring reconciliation against code.\n"
        "• INTERPRETATION: A reasoned engineering deduction derived from verified evidence.\n"
        "• HYPOTHESIS: A plausible scientific explanation that requires future empirical verification.\n"
        "• NOT VERIFIED: Currently unsupported by sufficient workspace artifacts or code.",
        "EVIDENCE"
    )

    add_h2(doc, "Forensic Reconciliation: Code Reality vs. Previous Study Notes")
    add_body(doc, "The project's baseline documentation, titled 'Federated_SSL_Project_Study_Notes-after 3rounds of training.docx' (approx. 32 pages), captured the preliminary design goals of the project. However, substantial engineering refactoring occurred between the initial 3-round prototype and the current 100-round production code. The table below forensicly reconciles the initial study notes with the verified repository state.")

    reconcile_headers = ["Subsystem / Concept", "Reported in 3-Round Notes", "Verified in Current Codebase (dev)", "Engineering Rationale & Impact"]
    reconcile_rows = [
        [
            "Primary Encoder Backbone",
            "ResNet50 (2048-dim features projected to 512-dim)",
            "ViT-Tiny (patch16_224, 192-dim features) via timm",
            "ViT-Tiny enables native patch-based MAE masking without ad-hoc convolutional feature pooling."
        ],
        [
            "Stage 1 Federated Sharing",
            "Only Encoder weights shared; Decoder stays strictly local",
            "Both Encoder and Decoder weights federated across rounds",
            "Reconstruction pretext learning collapses if hospital clients discard the global decoder each round."
        ],
        [
            "Prototypical Metric Function",
            "Euclidean Distance: -||z_query - c_k||^2",
            "Cosine Similarity with Temperature: 10.0 * (z · c) / (||z|| ||c||)",
            "Normalized cosine metric prevents gradient explosion and stabilizes few-shot cross-entropy optimization."
        ],
        [
            "Client Count & Non-IID Alpha",
            "5 Simulated Hospitals, Dirichlet alpha = 2.0",
            "10 Simulated Hospitals, Dirichlet alpha = 1.0 (default.yaml)",
            "10 clients represent higher realistic clinical fragmentation with sharper quantity skew (alpha=1.0)."
        ],
        [
            "Few-Shot Adaptation Mechanics",
            "Static centroid calculation without backprop or training",
            "Leave-one-out support adaptation with AdamW and clip_grad_norm",
            "Projection head adapts to tuberculosis feature manifolds using support episodes while keeping test sets isolated."
        ],
        [
            "Web Serving Framework",
            "Flask REST API serving static training_log.json",
            "FastAPI with Uvicorn, async endpoints, and CORS middleware",
            "High-throughput asynchronous image upload, OpenAPI docs, and real-time frontend status polling."
        ],
        [
            "UI Metrics Presentation",
            "Generic simulated progression table",
            "Calibrated 30-Round Shenzhen Evaluation: AUC=70.97%, Sens=79.15%",
            "Directly hardcoded in App.jsx (lines 36-43) to guarantee stable clinical presentation during live demonstrations."
        ],
        [
            "Cross-Dataset Generalization",
            "Assumed high diagnostic accuracy across all datasets",
            "Significant domain gap: ~81% Shenzhen AUC vs ~51% Montgomery AUC",
            "Reveals critical radiographic distribution shift between hospital populations; highlights need for domain adaptation."
        ]
    ]
    add_table(doc, reconcile_headers, reconcile_rows, [1.4, 1.7, 1.8, 1.6])

    add_body(doc, "Understanding these discrepancies is vital for viva examination and technical reviews. You must never claim in an examination that the model runs ResNet50 or computes Euclidean distances when the inspected codebase clearly runs ViT-Tiny and cosine metric scaling.", bold_prefix="Defense Imperative: ")

    add_horizontal_rule(doc)
