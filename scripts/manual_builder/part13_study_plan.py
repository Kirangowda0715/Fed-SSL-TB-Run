"""
scripts/manual_builder/part13_study_plan.py
-------------------------------------------
Part XIII: Structured 12-Week (4 Hours/Day) Engineering Mastery Programme
with Daily Objectives, Source File Anchors, and Verification Milestones.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part13(doc):
    add_h1(doc, "Part XIII: Structured 12-Week Study Programme (4h / Day)")

    add_h2(doc, "13.1 Programme Design & Progression Gateways")
    add_body(doc, "This curriculum is calibrated around 4 focused study hours per day, progressing from foundational Python/PyTorch fluency to independent repository modification, experimental reproduction, and academic viva defense.")

    add_body(doc, "Progression Gateway Rules:", bold_prefix="Strict Learning Rules: ")
    add_bullet(doc, "Never move to the next week without passing the practical verification task and answering all recall questions without notes.", bold_prefix="Rule 1 (Mastery Gate): ")
    add_bullet(doc, "Type out the core functions from memory at least once into a scratch file before proceeding.", bold_prefix="Rule 2 (Code Typing): ")
    add_bullet(doc, "Every experimental claim must be verified against actual JSON metrics or live command executions.", bold_prefix="Rule 3 (Evidence Verification): ")

    add_h2(doc, "13.2 Twelve-Week Syllabus Overview")

    curriculum = [
        ("Week 1: Python, PyTorch & Tensor Foundations",
         "Tensors, matrix multiplication, autograd computational graphs, Dataset & DataLoader mechanics.",
         "src/utils/reproducibility.py, test_device.py",
         "Build a custom PyTorch Dataset that loads synthetic tensors and trains a 2-layer MLP with AdamW.",
         "test_smoke.py sections 1-2 pass."),
        
        ("Week 2: Git & GitHub Workflow in this Project",
         "DAG commits, index staging, branch pointers, remote tracking (origin vs second), merge conflicts.",
         "git log, git status, git remote, .gitignore",
         "Create an experiment branch, stage a YAML modification, commit, resolve an intentional conflict, and inspect diffs.",
         "git branch -a -v and git log -n 5 output clean."),
        
        ("Week 3: Codebase Architecture & Orchestration",
         "Simulation harness, SimpleNamespace YAML parsing, server initialization, in-memory client simulation.",
         "src/utils/config.py, src/federated/simulation.py, src/server/server.py",
         "Trace the main() execution loop step-by-step using Python interactive debugger (pdb).",
         "python src/federated/simulation.py --dry-run completes successfully."),
        
        ("Week 4: Medical Data Pipeline & Partitioning",
         "Chest X-ray transforms, PIL to tensor, Dirichlet non-IID quantity skew, cache hashing and validation.",
         "src/datasets/loader.py, src/datasets/splitter.py",
         "Inspect raw Shenzhen and Montgomery folders; run splitter with alpha=0.5 and verify split_metadata.json.",
         "python -m unittest test_split_cache.py passes."),
        
        ("Week 5: Vision Transformers (ViT) Internals",
         "Patch extraction, linear projections, sinusoidal positional embeddings, Multi-Head Self-Attention math.",
         "src/models/encoder.py, timm integration",
         "Recreate ViTTinyEncoder from scratch in a scratchpad script; verify output shape is (B, 192).",
         "python -c 'from src.models.encoder import get_encoder; m=get_encoder(); print(m)' runs cleanly."),
        
        ("Week 6: Masked Autoencoders (MAE) Pretext Task",
         "Asymmetric encoder-decoder, random patch masking (80%), ids_restore shuffling, masked MSE loss.",
         "src/models/decoder.py, src/models/mae.py",
         "Trace a single batch of images through patchify, masking, decoder reconstruction, and loss calculation.",
         "python test_smoke.py sections 3-4 pass."),
        
        ("Week 7: Federated Optimization (FedAvg & FedProx)",
         "Distributed objective, client drift, FedProx proximal regularization autograd, sample-weighted averaging.",
         "src/client/ssl_train.py, src/server/aggregator.py",
         "Calculate a 3-client FedAvg aggregation manually on paper; compare against aggregator.py output.",
         "python -m unittest test_stage1_architecture.py passes."),
        
        ("Week 8: Full Federated Simulation & Checkpoint Lifecycle",
         "Round broadcast, local AdamW training, state_dict cloning, flame_round_XXX.pt checkpointing, resume logic.",
         "src/federated/simulation.py, src/server/server.py",
         "Run a 2-round synthetic simulation; interrupt it; execute --resume and verify it resumes from round 3.",
         "Checkpoint saved and loaded with matching round number."),
        
        ("Week 9: Few-Shot Metric Learning & Prototypical Networks",
         "Metric space theory, 2-layer projection MLP, temperature-scaled cosine similarity (tau=10.0), prototypes.",
         "src/models/proto_head.py, src/client/local_train.py",
         "Verify why K >= 2 is mandatory for leave-one-out adaptation; trace episode loss computation.",
         "python -m unittest test_few_shot.py passes."),
        
        ("Week 10: Clinical Diagnostic Metrics & Experiment Analysis",
         "ROC-AUC, sensitivity, specificity, F1, balanced accuracy, cross-dataset generalization gap.",
         "src/utils/metrics.py, experiments/ablation_studies/",
         "Reproduce the Round 49 vs Round 99 comparison table from raw JSON files; compute confusion matrices.",
         "Verify App.jsx hardcoded metrics match stage2_k5_round029.json."),
        
        ("Week 11: IEEE Paper Preparation & Claim Defense",
         "Paper structure, related work, claim verification matrix, addressing reviewer attacks, ethics.",
         "PROJECT_DOCUMENTATION.md, research_audit.py",
         "Draft an IEEE conference abstract and 4-paragraph methodology section based on verified source code.",
         "python research_audit.py reports zero hard failures."),
        
        ("Week 12: Viva Examination & Technical Placement Mastery",
         "35 comprehensive viva questions, mock technical interview, edge case debugging, coding challenges.",
         "FedSSL_TB_Engineering_Mastery_Manual.docx",
         "Conduct a 45-minute mock viva defense answering all questions without hesitation or notes.",
         "Complete all live coding tasks with 100% correct code.")
    ]

    for title, concepts, files, task, gate in curriculum:
        add_h3(doc, title)
        add_body(doc, concepts, bold_prefix="Core Concepts: ")
        add_body(doc, files, bold_prefix="Primary Source Files: ")
        add_body(doc, task, bold_prefix="Practical Exercise: ")
        add_body(doc, gate, bold_prefix="Mastery Gate: ")

    add_horizontal_rule(doc)
