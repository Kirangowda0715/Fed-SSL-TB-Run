"""
scripts/manual_builder/part9_experiments.py
-------------------------------------------
Part IX: Empirical Experiment Registry, 30-Round and 100-Round Runs,
Controlled Ablation Studies, and Cross-Dataset Generalization Analysis.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part9(doc):
    add_h1(doc, "Part IX: Experiments — What We Ran & What We Learned")

    add_h2(doc, "9.1 Consolidated Experiment Registry")
    add_body(doc, "Every experimental result discussed in this manual is grounded in concrete saved artifacts in the repository. The registry below catalogs every verified run:")

    reg_headers = ["Exp ID", "Description & Target", "Checkpoint / Log Path", "FL Rounds", "Few-Shot K", "Encoder Mode", "Verification Status"]
    reg_rows = [
        ["EXP-30R-K5", "30-Round Baseline (K=5)", "experiments/ablation_30_round/metrics/stage2_k5_round029.json", "30", "5", "Frozen", "OBSERVED EXPERIMENT"],
        ["EXP-30R-K20", "30-Round Baseline (K=20)", "experiments/ablation_30_round/metrics/stage2_k20_round029.json", "30", "20", "Frozen", "OBSERVED EXPERIMENT"],
        ["EXP-100R-R49", "Midpoint 100-Round (Round 49)", "experiments/ablation_studies/round49_vs_round99/round049_results.json", "50", "20", "Frozen", "OBSERVED EXPERIMENT"],
        ["EXP-100R-R99", "Final 100-Round (Round 99)", "experiments/ablation_studies/round49_vs_round99/round099_results.json", "100", "20", "Frozen", "OBSERVED EXPERIMENT"],
        ["ABL-K-SHOT", "Few-Shot K Sweep (5..50)", "experiments/ablation_studies/few_shot_k/results.json", "100", "5,10,15,20,30,50", "Frozen", "OBSERVED EXPERIMENT"],
        ["ABL-FREEZE", "Frozen vs Unfrozen Adaptation", "experiments/ablation_studies/frozen_vs_unfrozen/results.json", "100", "20", "Frozen vs Trainable", "OBSERVED EXPERIMENT"]
    ]
    add_table(doc, reg_headers, reg_rows, [1.1, 1.6, 1.8, 0.6, 0.7, 0.7, 1.0])

    add_h2(doc, "9.2 The 30-Round Baseline & Calibrated Dashboard Metrics")
    add_body(doc, "The 30-round federated training run served as the primary proof-of-concept for the web dashboard. Evaluated at Round 29 with K=5 support samples, the metrics achieved were:")

    m30_headers = ["Evaluation Cohort", "ROC-AUC", "Accuracy", "Sensitivity", "Specificity", "F1-Score", "Balanced Accuracy"]
    m30_rows = [
        ["Shenzhen Query Set (In-Domain)", "0.7097 (70.97%)", "0.6733 (67.33%)", "0.7915 (79.15%)", "0.5514 (55.14%)", "0.7110 (71.10%)", "0.6715 (67.15%)"],
        ["Montgomery Test Set (Out-of-Domain)", "0.5569 (55.69%)", "0.4420 (44.20%)", "1.0000 (100.0%)", "0.0375 (3.75%)", "0.6010 (60.10%)", "0.5188 (51.88%)"]
    ]
    add_table(doc, m30_headers, m30_rows, [1.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    add_body(doc, "Direct Code Provenance in App.jsx:", bold_prefix="UI Consistency Anchor: ")
    add_body(doc, "The Shenzhen metrics above (AUC = 0.709659, Acc = 0.673313, Sens = 0.791541, Spec = 0.551402, F1 = 0.710991) are the EXACT numbers hardcoded in src/web/frontend/src/App.jsx (lines 36-43) to guarantee that the web dashboard presents authentic, calibrated experimental findings without risk of broken network links during presentations.")

    add_h2(doc, "9.3 100-Round Pretraining Progression: Round 49 vs. Round 99")
    add_body(doc, "To investigate whether extending federated SSL pretraining from 50 rounds to 100 rounds improves downstream clinical accuracy, we executed a controlled comparison holding all Stage 2 adaptation variables strictly identical (Seed=42, K=20, Projection Dim=128, Adaptation Epochs=50):")

    comp_headers = ["Evaluation Cohort", "Metric", "Round 49 Checkpoint", "Round 99 Checkpoint", "Delta (R99 - R49)", "Scientific Takeaway"]
    comp_rows = [
        ["Shenzhen (In-Domain)", "ROC-AUC", "0.8142", "0.8126", "-0.0016 (-0.16%)", "Pretraining reached representation saturation by Round 49."],
        ["Shenzhen (In-Domain)", "Accuracy", "0.7685", "0.7749", "+0.0064 (+0.64%)", "Marginal accuracy improvement within seed variance."],
        ["Shenzhen (In-Domain)", "Sensitivity", "0.7595", "0.7848", "+0.0253 (+2.53%)", "TB recall improved by 2.5%."],
        ["Shenzhen (In-Domain)", "Specificity", "0.7778", "0.7647", "-0.0131 (-1.31%)", "Normal specificity slightly traded for recall."],
        ["Shenzhen (In-Domain)", "F1-Score", "0.7692", "0.7799", "+0.0107 (+1.07%)", "Overall F1 increased by 1.1%."],
        ["Montgomery (Held-Out)", "ROC-AUC", "0.5248", "0.5129", "-0.0119 (-1.19%)", "Out-of-domain performance hovered near random chance."],
        ["Montgomery (Held-Out)", "Accuracy", "0.5145", "0.4783", "-0.0362 (-3.62%)", "Cross-dataset classification degraded slightly."]
    ]
    add_table(doc, comp_headers, comp_rows, [1.4, 1.0, 1.2, 1.2, 1.1, 1.6])

    add_h2(doc, "9.4 Controlled Ablation Studies")
    add_h3(doc, "Ablation Study 1: Impact of Support Set Size K (Few-Shot Sweep)")
    add_body(doc, "We evaluated few-shot support sizes K in {5, 10, 15, 20, 30, 50} on Round 99 to observe metric sensitivity:")

    k_headers = ["K-Shot per Class", "Total Support Size", "Shenzhen AUC", "Shenzhen Accuracy", "Montgomery AUC", "Montgomery Accuracy"]
    k_rows = [
        ["K = 5", "10 images (5 Normal + 5 TB)", "0.7153", "0.6595", "0.6365", "0.6087"],
        ["K = 10", "20 images (10 Normal + 10 TB)", "0.7928", "0.7165", "0.5550", "0.5507"],
        ["K = 15", "30 images (15 Normal + 15 TB)", "0.7470", "0.6994", "0.5878", "0.5652"],
        ["K = 20", "40 images (20 Normal + 20 TB)", "0.8126", "0.7749", "0.5129", "0.4783"],
        ["K = 30", "60 images (30 Normal + 30 TB)", "0.8059", "0.6977", "0.4968", "0.4783"],
        ["K = 50", "100 images (50 Normal + 50 TB)", "0.8016", "0.7456", "0.5511", "0.5000"]
    ]
    add_table(doc, k_headers, k_rows, [1.2, 1.8, 1.1, 1.1, 1.1, 1.1])

    add_body(doc, "Scientific Finding: On Shenzhen, AUC climbs sharply from 0.7153 at K=5 to peak at 0.8126 at K=20. However, on Montgomery, K=5 achieved the highest cross-dataset AUC (0.6365). As K increases, the support prototypes fit Shenzhen-specific scanner characteristics more tightly, exacerbating domain transfer penalty on Montgomery.", bold_prefix="Key Takeaway: ")

    add_h3(doc, "Ablation Study 2: Frozen vs. Unfrozen Encoder Adaptation")
    add_body(doc, "We tested adapting the full ViT-Tiny encoder during Stage 2 vs. freezing it and training only the projection head:")

    frz_headers = ["Encoder Mode", "Trainable Parameters", "Shenzhen AUC", "Shenzhen Accuracy", "Montgomery AUC", "Montgomery Accuracy"]
    frz_rows = [
        ["Frozen Encoder", "Projection Head only (~41k params)", "0.8126", "0.7749", "0.5129", "0.4783"],
        ["Unfrozen Encoder", "Encoder + Head (~5.5M params)", "0.7989", "0.7219", "0.5461", "0.5290"]
    ]
    add_table(doc, frz_headers, frz_rows, [1.4, 1.8, 1.1, 1.1, 1.1, 1.1])

    add_h2(doc, "9.5 The Generalization Gap: Clinical & Research Analysis")
    add_callout(doc, "HOW TO DEFEND THE SHENZHEN VS. MONTGOMERY GENERALIZATION GAP",
        "Viva Examination Defense: Why does the model achieve ~81% AUC on Shenzhen, but drops to ~51-55% AUC on Montgomery?\n\n"
        "1. Radiographic Domain Shift: Shenzhen radiographs were collected using a stationary digital radiography system at Shenzhen No. 3 Hospital in China, whereas Montgomery images were acquired using a distinct rotary tube system in Maryland, USA. The two datasets have vastly different mean intensities, contrast ratios, and thoracic aspect ratios.\n"
        "2. Prototype Entanglement: The class prototypes c_Normal and c_TB are computed from Shenzhen images. When Montgomery images are projected into the same metric space, scanner-specific feature shifts displace Montgomery embeddings, causing them to fall closer to the wrong prototype.\n"
        "3. Research Integrity: This is an authentic, published medical imaging phenomenon (cross-site dataset shift). Acknowledging this discrepancy demonstrates mature scientific understanding, whereas claiming 95% generalization without domain adaptation would be immediately rejected by an IEEE reviewer or expert examiner.",
        "PROFESSOR"
    )

    add_horizontal_rule(doc)
