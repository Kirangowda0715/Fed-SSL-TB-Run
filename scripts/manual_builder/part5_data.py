"""
scripts/manual_builder/part5_data.py
-----------------------------------
Part V: Medical Data Pipeline, Augmentations, Splitting Mathematics,
and Leakage Prevention Protocols.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part5(doc):
    add_h1(doc, "Part V: Understand the Data Pipeline & Correctness")

    add_h2(doc, "5.1 The Three Medical Datasets")
    add_body(doc, "The project relies on three distinct radiographic cohorts, each serving an isolated and strictly defined role in the machine learning lifecycle:")

    ds_headers = ["Dataset Name", "Source Institution & Geography", "Total Size / Used", "Label Status", "Role in Pipeline"]
    ds_rows = [
        [
            "NIH ChestX-ray14",
            "National Institutes of Health Clinical Center (Maryland, USA)",
            "112,120 total / 500-20,000 used",
            "Unlabeled (Labels discarded)",
            "Stage 1: Decentralized self-supervised MAE pretraining across 10 hospital shards."
        ],
        [
            "Shenzhen Hospital TB Dataset",
            "Shenzhen No. 3 People's Hospital (Guangdong, China)",
            "662 images (326 TB, 336 Normal)",
            "Binary Gold-Standard (TB=1, Normal=0)",
            "Stage 2: Support set extraction for few-shot metric adaptation (K=5 to 20 samples)."
        ],
        [
            "Montgomery County TB Dataset",
            "Montgomery County Department of Health (Maryland, USA)",
            "138 images (58 TB, 80 Normal)",
            "Binary Gold-Standard (TB=1, Normal=0)",
            "Stage 3: Strictly held-out external evaluation set for cross-hospital testing."
        ]
    ]
    add_table(doc, ds_headers, ds_rows, [1.5, 1.8, 1.2, 1.0, 1.5])

    add_h2(doc, "5.2 Image Transformations & Radiographic Preprocessing")
    add_body(doc, "Medical chest X-rays vary wildly in resolution (ranging from 1024x1024 up to 3000x3000) and bit depth. The pipeline standardizes every radiograph through an explicit sequence of transforms in src/datasets/loader.py:")

    trans_headers = ["Transform Step", "PyTorch Function", "Transformation Effect", "Clinical / Engineering Justification"]
    trans_rows = [
        ["1. Spatial Rescaling", "T.Resize((256, 256))", "Scales high-res X-ray down to 256x256 pixels.", "Standardizes arbitrary clinical resolutions while retaining gross thoracic anatomy."],
        ["2. Data Augmentation", "T.RandomCrop(224)", "Crops random 224x224 window during training.", "Simulates minor patient positioning shifts and prevents memorization."],
        ["3. Anatomical Flip", "T.RandomHorizontalFlip(p=0.5)", "Flips left and right hemithorax horizontally.", "Bilateral lung symmetry: TB can manifest in either apex; doubles effective training diversity."],
        ["4. Channel Expansion", "T.Grayscale(num_output_channels=3)", "Duplicates single X-ray channel to 3 channels.", "Chest X-rays are 1-channel grayscale; duplicating channels allows native use of 3-channel ViT architectures."],
        ["5. Tensor Conversion", "T.ToTensor()", "Converts PIL image [0, 255] to torch.float32 [0.0, 1.0].", "Converts image pixels to PyTorch tensors with shape (3, 224, 224)."],
        ["6. Standard Normalization", "T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)", "Applies (x - mean) / std per channel.", "Standardizes input distribution around zero mean, accelerating transformer convergence."]
    ]
    add_table(doc, trans_headers, trans_rows, [1.3, 1.7, 1.7, 1.8])

    add_h2(doc, "5.3 Non-IID Partitioning Mathematics: Dirichlet Quantity Skew")
    add_body(doc, "In real-world healthcare federations, different hospitals have vastly different patient throughput. A large regional medical center might scan 500 patients per day, while a community health outpost scans only 15. This is simulated in src/datasets/splitter.py using a Dirichlet distribution:")

    add_body(doc, "Mathematical Formulation:", bold_prefix="Dirichlet Distribution: ")
    add_body(doc, "Let K be the number of participating hospitals (K=10). We sample a probability vector p = (p_1, p_2, ..., p_K) from a symmetric Dirichlet distribution with concentration parameter alpha:")

    add_code_block(doc, "p ~ Dirichlet(alpha * 1_K) = (1 / B(alpha)) * prod_{k=1}^K p_k^(alpha - 1)")

    add_body(doc, "Where alpha controls the level of statistical quantity heterogeneity across nodes:")
    add_bullet(doc, "When alpha -> infinity: p_k = 1/K for all k. Every hospital receives an identical, uniform number of images (IID quantity distribution).", bold_prefix="alpha -> infinity (Homogeneous): ")
    add_bullet(doc, "When alpha = 1.0 (Our Default): Sample proportions follow a uniform distribution over the K-1 dimensional simplex. Generates realistic moderate quantity skew across hospital nodes.", bold_prefix="alpha = 1.0 (Realistic Skew): ")
    add_bullet(doc, "When alpha -> 0: Extreme concentration where one or two hospitals receive almost all the data, while others receive nearly zero samples.", bold_prefix="alpha -> 0 (Severe Skew): ")

    add_body(doc, "In src/datasets/splitter.py, to guarantee that every hospital receives at least one sample, the implementation initializes each node with 1 sample and allocates the remaining N - K samples via a Multinomial draw:")

    code_split = (
        "def _split_non_iid(indices: np.ndarray, num_hospitals: int, alpha: float) -> List[np.ndarray]:\n"
        "    n = len(indices)\n"
        "    np.random.shuffle(indices)\n"
        "    # Sample hospital proportions from Dirichlet\n"
        "    proportions = np.random.dirichlet(alpha=np.ones(num_hospitals) * alpha)\n"
        "    # Ensure non-empty partitions: 1 base sample + Multinomial allocation\n"
        "    counts = np.ones(num_hospitals, dtype=int)\n"
        "    counts += np.random.multinomial(n - num_hospitals, proportions)\n"
        "    # Slice disjoint subsets\n"
        "    hospital_indices = []\n"
        "    start = 0\n"
        "    for count in counts:\n"
        "        hospital_indices.append(indices[start : start + count])\n"
        "        start += count\n"
        "    return hospital_indices"
    )
    add_code_block(doc, code_split)

    add_h2(doc, "5.4 Cache Validation & Data Leakage Prevention")
    add_body(doc, "A common flaw in academic federated learning implementations is partition cache staleness: changing the dataset size, hospital count, or random seed in YAML while the simulation silently reuses old indices from disk. Our repository enforces rigorous cache validation via split_cache_matches():")

    add_bullet(doc, "Every partition write persists data/processed/split_metadata.json storing exact parameters: dataset_size, num_hospitals, strategy, alpha, and seed.", bold_prefix="1. Metadata Hashing: ")
    add_bullet(doc, "On startup, split_cache_matches() verifies that the cached metadata exactly equals the current runtime configuration. If any value differs, the cache is automatically invalidated and recomputed.", bold_prefix="2. Automated Invalidation: ")
    add_bullet(doc, "In evaluate_on_montgomery(), the code resolves absolute file paths and executes an explicit set intersection check: assert not (support_paths & test_paths). If a single image appears in both Shenzhen support and Montgomery test sets, execution halts immediately with a fatal ValueError.", bold_prefix="3. Absolute Leakage Barrier: ")

    add_horizontal_rule(doc)
