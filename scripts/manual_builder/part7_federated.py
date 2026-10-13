"""
scripts/manual_builder/part7_federated.py
-----------------------------------------
Part VII: Federated Optimization Mechanics: FedAvg, FedProx,
In-Memory Simulation Architecture, and 3-Client Numerical Worked Example.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part7(doc):
    add_h1(doc, "Part VII: Federated Optimization (FedAvg & FedProx)")

    add_h2(doc, "7.1 Federated Optimization Formulations")
    add_body(doc, "In standard centralized learning, all data is collected into a central server to solve min_w F(w) = (1/N) sum_{i=1}^N L(x_i; w). In federated healthcare learning, data is partitioned across K clinical nodes, each holding n_k local radiographs:")

    add_code_block(doc, "min_w f(w) = sum_{k=1}^K (n_k / N) F_k(w)\n\nwhere F_k(w) = (1 / n_k) sum_{i in D_k} L_MAE(x_i; w)")

    add_h3(doc, "Federated Averaging (FedAvg; McMahan et al., AISTATS 2017)")
    add_body(doc, "In FedAvg, each client initializes local weights with the broadcast global model w^t, executes E local training epochs, and transmits updated weights w_k^(t+1) back to the server. The server aggregates updates via sample-weighted averaging:")

    add_code_block(doc, "w^(t+1) = sum_{k=1}^K (n_k / N) w_k^(t+1)")

    add_h3(doc, "FedProx Optimization (Li et al., MLSys 2020)")
    add_body(doc, "When local hospital datasets are non-IID (heterogeneous quantity and pathology distributions), client optimization trajectories diverge widely. This phenomenon, known as Client Drift, causes FedAvg aggregation to oscillate or diverge.")

    add_body(doc, "FedProx adds a proximal regularization term to each hospital client's local loss function, penalizing local parameters that drift too far from the global consensus model w^t:")

    add_code_block(doc, "min_w h_k(w; w^t) = F_k(w) + (mu / 2) ||w - w^t||_2^2\n\nwhere mu = 0.01 in configs/default.yaml")

    add_callout(doc, "CRITICAL IMPLEMENTATION INSIGHT: WHERE DOES FEDPROX LIVE?",
        "Viva Examination Defense: Where is the FedProx algorithm implemented in code?\n\n"
        "• On the Client Side (src/client/ssl_train.py): The proximal term (mu/2) * ||w_local - w_global||^2 is computed directly on the active PyTorch autograd graph and added to the MAE loss before loss.backward(). This modifies local client gradients during backpropagation.\n"
        "• On the Server Side (src/server/aggregator.py): FedProx uses the EXACT SAME weighted averaging as FedAvg! The server does not apply additional proximal regularization during aggregation because the stabilization was already baked into each client's local updates.",
        "EVIDENCE"
    )

    add_h2(doc, "7.2 Fully Worked 3-Client Numerical Aggregation Example")
    add_body(doc, "To verify the exact mathematics executed in src/server/aggregator.py, consider a concrete numerical example with 3 hospitals aggregating a 1-parameter weight tensor w:")

    add_body(doc, "Scenario Setup:", bold_prefix="Setup: ")
    add_bullet(doc, "Hospital 1 has n_1 = 200 samples. After local training, its parameter is w_1 = 0.450.", bold_prefix="Hospital 1: ")
    add_bullet(doc, "Hospital 2 has n_2 = 300 samples. After local training, its parameter is w_2 = 0.520.", bold_prefix="Hospital 2: ")
    add_bullet(doc, "Hospital 3 has n_3 = 500 samples. After local training, its parameter is w_3 = 0.610.", bold_prefix="Hospital 3: ")

    calc_steps = (
        "Step 1: Compute Total Sample Count N\n"
        "   N = n_1 + n_2 + n_3 = 200 + 300 + 500 = 1000 total samples\n"
        "\n"
        "Step 2: Calculate Normalized Hospital Aggregation Weights\n"
        "   alpha_1 = n_1 / N = 200 / 1000 = 0.200 (20% influence)\n"
        "   alpha_2 = n_2 / N = 300 / 1000 = 0.300 (30% influence)\n"
        "   alpha_3 = n_3 / N = 500 / 1000 = 0.500 (50% influence)\n"
        "   Check: sum(alpha_k) = 0.200 + 0.300 + 0.500 = 1.000 (Proper convex combination)\n"
        "\n"
        "Step 3: Execute Convex Weighted Parameter Aggregation\n"
        "   w_global = alpha_1 * w_1 + alpha_2 * w_2 + alpha_3 * w_3\n"
        "   w_global = (0.200 * 0.450) + (0.300 * 0.520) + (0.500 * 0.610)\n"
        "   w_global = 0.0900 + 0.1560 + 0.3050\n"
        "   w_global = 0.5510\n"
        "\n"
        "Step 4: Update Global Server Model State\n"
        "   The global server model updates its parameter to exactly 0.5510 and saves to checkpoint."
    )
    add_code_block(doc, calc_steps)

    add_body(doc, "What Happens to Non-Floating State Dict Buffers?", bold_prefix="Edge Case Handling: ")
    add_body(doc, "In deep learning models, certain state_dict entries are integers (e.g., num_batches_tracked in BatchNorm). Floating-point weighted averaging of integer buffers results in dtype corruption. In src/server/aggregator.py, lines 64-67 explicitly check if parameter.dtype.is_floating_point. If false, the buffer is cloned directly from client 0 without averaging, guaranteeing type stability.")

    add_h2(doc, "7.3 In-Memory Simulation Mechanics")
    add_body(doc, "In academic federated learning research (such as the LEAF benchmark and FedML), running 10 separate physical machines or Docker containers across a local network introduces unnecessary network latency and hardware overhead. FedSSL-TB uses an in-memory simulation harness:")

    add_bullet(doc, "All 10 client models are simulated sequentially (or in multi-threaded parallel across GPUs) within a single Python runtime using copy.deepcopy(global_model).", bold_prefix="1. Deepcopy Isolation: ")
    add_bullet(doc, "Memory hygiene: After a client completes training and transmits its state_dict, the client model instance is discarded, freeing GPU VRAM before the next hospital client loads.", bold_prefix="2. VRAM Reclamation: ")
    add_bullet(doc, "Multi-GPU Acceleration: The --parallel flag invokes Python's ThreadPoolExecutor, mapping hospital workers across available CUDA devices (e.g., cuda:0, cuda:1).", bold_prefix="3. Parallel Worker Scheduling: ")

    add_horizontal_rule(doc)
