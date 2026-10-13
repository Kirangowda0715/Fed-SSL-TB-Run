"""
scripts/manual_builder/part3_git.py
-----------------------------------
Part III: Practical Git and GitHub Engineering Workflow Grounded in Real Repository Evidence.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part3(doc):
    add_h1(doc, "Part III: Git & GitHub Workflow in this Actual Project")

    add_h2(doc, "3.1 Git Mental Model Grounded in FedSSL-TB")
    add_body(doc, "Git is a content-addressable directed acyclic graph (DAG) of project snapshots. Unlike traditional Version Control Systems (VCS) that store file differences (deltas), Git stores full snapshots of the tree structure indexed by SHA-1 hashes.")

    add_body(doc, "The repository operates across four distinct operational zones:", bold_prefix="The Four Operational Zones: ")
    add_bullet(doc, "d:\\Projects\\1.Final Year Project\\Federated-SSL — Where you edit Python files, run scripts, and view local datasets.", bold_prefix="1. Working Directory: ")
    add_bullet(doc, "The Index (Staging Area) — A binary cache (.git/index) holding changes scheduled for the next commit snapshot.", bold_prefix="2. Staging Area: ")
    add_bullet(doc, ".git/ directory on disk — The local DAG holding all commit objects, tree objects, and blob objects.", bold_prefix="3. Local Repository: ")
    add_bullet(doc, "The GitHub remotes (origin and second) — Remote bare repositories synchronizing development across contributors.", bold_prefix="4. Remote Repositories: ")

    add_h2(doc, "3.2 Real Repository Remote & Branch Configuration")
    add_body(doc, "Our repository inspection establishes the exact active network configuration:", bold_prefix="VERIFIED IN EXECUTION: ")

    git_remotes = (
        "$ git remote -v\n"
        "origin  https://github.com/nmit-1NT23CS111/Fed-SSL-TB.git (fetch)\n"
        "origin  https://github.com/nmit-1NT23CS111/Fed-SSL-TB.git (push)\n"
        "second  https://github.com/Kirangowda0715/Fed-SSL-TB-Run.git (fetch)\n"
        "second  https://github.com/Kirangowda0715/Fed-SSL-TB-Run.git (push)\n"
        "\n"
        "$ git branch -a -v\n"
        "* dev                     a6354aa Add federated SSL ablation study experiments\n"
        "  main                    2e320fd [behind 1] frontend update\n"
        "  remotes/origin/main     2e320fd frontend update\n"
        "  remotes/origin/dev      803a61d Add evaluation metrics to training log for dashboard rendering\n"
        "  remotes/second/dev      a6354aa Add federated SSL ablation study experiments\n"
        "  remotes/second/main     eb13a9e Update prototypical head and fix metadata loader"
    )
    add_code_block(doc, git_remotes)

    add_body(doc, "Architectural Takeaway: The active branch is dev, which tracks second/dev at commit a6354aa. The origin remote represents the upstream team/college repository, while second represents the personal active deployment repository used for cloud runs and ablation experiments.")

    add_h2(doc, "3.3 Verified Commit History & Feature Evolution")
    add_body(doc, "Inspecting the recent git log graph reveals how the project advanced from initial round logging to 100-round continuation and ablation studies:")

    commit_headers = ["Commit SHA", "Commit Message", "Author & Context", "Impact on Codebase"]
    commit_rows = [
        ["a6354aa", "Add federated SSL ablation study experiments", "Current HEAD (dev)", "Integrated controlled ablation runs for few_shot_k and round49_vs_round99."],
        ["5e11b68", "Improve README setup and project documentation", "Documentation Update", "Standardized setup commands and project overview."],
        ["8b436ca", "experiment folder organized", "Refactoring", "Cleaned up experiment subdirectories and result metrics."],
        ["32f66e9", "Clean up redundant project artifacts", "Cleanup", "Removed scratch files and outdated temporary files."],
        ["b5fc27d", "Update training logs and experiment results", "Results Update", "Synchronized training_log.json with 30-round run metrics."],
        ["117f7c9", "Add Round 99 federated checkpoint", "Model Weight Tracking", "Added flame_round_099.pt checkpoint from Kaggle GPU 100-round execution."],
        ["2f6bdcd", "Add round 49 checkpoint for 100 round continuation", "Continuation", "Added intermediate checkpoint flame_round_049.pt."],
        ["561a247", "Added checkpoints and created folder structure named suhas", "Kaggle Run Setup", "Configured Suhas Kaggle GPU environment to train from round 50 to 100."]
    ]
    add_table(doc, commit_headers, commit_rows, [1.1, 2.3, 1.4, 1.7])

    add_h2(doc, "3.4 Concrete Git Command Runbook for Daily Engineering")
    add_body(doc, "Below is the exact step-by-step command reference for operating safely within this repository without risking data loss or merge conflicts.")

    cmd_headers = ["Action", "Exact Terminal Command", "State Before Execution", "What Git Does Internally", "Verification Command"]
    cmd_rows = [
        [
            "Check Repository Status",
            "git status",
            "Any state in working tree",
            "Scans working tree against index; reports staged, modified, and untracked files.",
            "Visual output review"
        ],
        [
            "Stage Code Changes",
            "git add src/ configs/",
            "Modified .py or .yaml files in working tree",
            "Calculates SHA-1 of files, writes blobs to .git/objects, updates index table.",
            "git diff --staged"
        ],
        [
            "Commit Snapshot",
            "git commit -m 'Descriptive message'",
            "Files staged in index",
            "Creates tree object from index, creates commit object pointing to parent commit, moves branch ref.",
            "git log -n 1 --oneline"
        ],
        [
            "Switch to dev Branch",
            "git switch dev",
            "On any other branch with clean tree",
            "Updates HEAD to refs/heads/dev, rewrites working directory files to match dev tree.",
            "git branch --show-current"
        ],
        [
            "Fetch Remote Updates",
            "git fetch second",
            "Local dev branch",
            "Downloads commits and trees from second remote; updates refs/remotes/second/* without touching working tree.",
            "git log dev..second/dev"
        ],
        [
            "Push to Personal Remote",
            "git push second dev",
            "Local dev has commits ahead of second/dev",
            "Transfers packfile of missing objects to GitHub; advances refs/heads/dev on remote.",
            "git status (shows up to date)"
        ],
        [
            "Create Experiment Branch",
            "git switch -c experiment/few-shot-k",
            "On dev branch at commit a6354aa",
            "Creates new ref refs/heads/experiment/few-shot-k pointing to a6354aa; moves HEAD to new ref.",
            "git branch -a"
        ],
        [
            "Discard Unstaged Edits",
            "git restore <file>",
            "Accidental edits in working directory",
            "Overwrites working tree file with the clean copy stored in the index.",
            "git status (clean)"
        ],
        [
            "Undo Accidental Commit",
            "git reset --soft HEAD~1",
            "Just made a commit with wrong message or extra files",
            "Moves branch reference back one commit; leaves all committed changes staged in the index.",
            "git status (shows staged files)"
        ]
    ]
    add_table(doc, cmd_headers, cmd_rows, [1.3, 1.6, 1.2, 1.4, 1.0])

    add_h2(doc, "3.5 Resolving Merge Conflicts Safely")
    add_body(doc, "A merge conflict occurs when two branches modify the same lines in a file differently. Here is the exact recovery procedure:")
    add_bullet(doc, "Run git status to identify conflicting files marked with 'both modified'.", bold_prefix="Step 1: Identify: ")
    add_bullet(doc, "Open the file. Look for conflict markers: <<<<<<< HEAD (your version), ======= (divider), and >>>>>>> second/dev (incoming version).", bold_prefix="Step 2: Inspect: ")
    add_bullet(doc, "Edit the code to keep the correct implementation. Delete all <<<<<<<, =======, and >>>>>>> markers.", bold_prefix="Step 3: Resolve: ")
    add_bullet(doc, "Stage the resolved file using git add <file>.", bold_prefix="Step 4: Stage: ")
    add_bullet(doc, "Complete the merge using git commit -m 'Resolve merge conflict between local and remote dev'.", bold_prefix="Step 5: Finalize: ")

    add_h2(doc, "3.6 Scientific Provenance vs. Git Commits")
    add_callout(doc, "SCIENTIFIC EXPERIMENT PROVENANCE REQUIREMENT",
        "A critical mistake in machine learning research is confusing a Git commit with an experiment result. A Git commit tracks code; it does NOT guarantee reproducible results alone.\n\n"
        "To establish true scientific provenance, every saved model checkpoint (flame_round_XXX.pt) in our project stores a metadata dictionary linking:\n"
        "1. Exact Git Commit SHA (e.g., a6354aa)\n"
        "2. Exact YAML Configuration Snapshot (model backbone, mask ratio, learning rate, FedProx mu)\n"
        "3. Pseudo-random Seed (seed=42 for PyTorch, CUDA, NumPy, and Python random)\n"
        "4. Hardware and CUDA Device Spec (e.g., NVIDIA T4 / P100 / RTX)\n"
        "5. Dataset Partition Hash (split_metadata.json hash for NIH Dirichlet split)\n\n"
        "This allows any researcher to reproduce the exact checkpoint from scratch without guessing what code state was used.",
        "NOTE"
    )

    add_horizontal_rule(doc)
