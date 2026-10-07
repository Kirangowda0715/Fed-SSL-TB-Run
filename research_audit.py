"""Consolidated research audit for FedSSL-TB.

Diagnostic only: no model training and no checkpoint modification.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import torch
from src.datasets.loader import ShenzhenDataset, MontgomeryDataset, get_eval_transform
from src.utils.config import load_config

PASS, WARN, FAIL = "PASS", "WARNING", "FAIL"

class Audit:
    def __init__(self): self.rows=[]
    def add(self, area, status, msg):
        self.rows.append((area,status,msg)); print(f"{status:<8} {area:<16} {msg}")
    def summary(self):
        c={PASS:0,WARN:0,FAIL:0}
        for _,s,_ in self.rows: c[s]+=1
        print("\n"+"="*72+"\nAUDIT SUMMARY\n"+"="*72)
        for k in (PASS,WARN,FAIL): print(f"{k:<9}: {c[k]}")
        result = "RESEARCH BLOCKERS FOUND" if c[FAIL] else ("NO HARD FAILURES, WARNINGS REQUIRE REVIEW" if c[WARN] else "ALL CHECKS PASSED")
        print(f"RESULT    : {result}")
        return c

def dataset_audit(name, ds, audit):
    if not len(ds): audit.add(name,FAIL,"dataset is empty"); return
    labels=ds.get_labels()
    paths=[str(Path(p).resolve()) for p in ds.image_paths]
    ids=list(ds.study_ids)
    audit.add(name,PASS,f"{len(ds)} images; Normal={labels.count(0)}, TB={labels.count(1)}")
    audit.add(name,PASS if len(paths)==len(set(paths)) else FAIL,"no duplicate image paths" if len(paths)==len(set(paths)) else "duplicate image paths detected")
    audit.add(name,PASS if len(ids)==len(set(ids)) else FAIL,"no duplicate study/image IDs" if len(ids)==len(set(ids)) else "duplicate study/image IDs detected")
    audit.add(name,PASS if set(labels)=={0,1} else FAIL,"binary labels contain both classes" if set(labels)=={0,1} else f"unexpected labels: {sorted(set(labels))}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml"); args=ap.parse_args()
    print("="*72+"\nFedSSL-TB CONSOLIDATED RESEARCH AUDIT\nDiagnostic only: no training/checkpoint modification\n"+"="*72)
    audit=Audit(); config=load_config(args.config)
    print("\n[DATASET INTEGRITY]")
    try:
        t=get_eval_transform(config.data.image_size)
        s=ShenzhenDataset(config.data.shenzhen_path,transform=t,image_size=config.data.image_size)
        m=MontgomeryDataset(config.data.montgomery_path,transform=t,image_size=config.data.image_size)
        dataset_audit("Shenzhen",s,audit); dataset_audit("Montgomery",m,audit)
        sp={str(Path(p).resolve()) for p in s.image_paths}; mp={str(Path(p).resolve()) for p in m.image_paths}
        si=set(s.study_ids); mi=set(m.study_ids)
        audit.add("Leakage",FAIL,f"{len(sp&mp)} Shenzhen/Montgomery image paths overlap" if sp&mp else "no Shenzhen/Montgomery image-path overlap")
        audit.add("Leakage",FAIL,f"{len(si&mi)} Shenzhen/Montgomery study IDs overlap" if si&mi else "no Shenzhen/Montgomery study-ID overlap")
    except Exception as e: audit.add("Dataset",FAIL,f"dataset audit failed: {e}")

    print("\n[STAGE 1 CONTRACTS]")
    r=subprocess.run([sys.executable,"-m","unittest","test_stage1_architecture.py","-q"],capture_output=True,text=True)
    audit.add("Stage 1",PASS,"architecture unit tests passed" if r.returncode==0 else FAIL,"")
    if r.returncode!=0: print(r.stdout,r.stderr)

    print("\n[CHECKPOINTS]")
    ckdir=Path(config.logging.checkpoint_dir); ckpts=sorted(ckdir.glob("*.pt"))
    audit.add("Checkpoints",PASS,f"{len(ckpts)} checkpoint files found" if ckpts else WARN,"")
    for p in ckpts:
        try:
            c=torch.load(p,map_location="cpu"); req={"encoder_state_dict","decoder_state_dict","proto_head_state_dict","config"}; missing=req-set(c)
            audit.add("Checkpoint",FAIL,f"{p.name} missing {sorted(missing)}" if missing else PASS and f"{p.name}: encoder={len(c['encoder_state_dict'])}, decoder={len(c['decoder_state_dict'])}, proto={len(c['proto_head_state_dict'])}")
        except Exception as e: audit.add("Checkpoint",FAIL,f"{p.name} unreadable: {e}")

    print("\n[STAGE 2 / EXPERIMENTS]")
    k=int(config.finetuning.few_shot_k)
    audit.add("Stage 2",PASS,f"K={k}; leave-one-out adaptation supported" if k>=2 else FAIL,"")
    audit.add("Stage 2",PASS,"encoder frozen during adaptation" if bool(config.finetuning.freeze_encoder) else WARN,"")
    logdir=Path(config.logging.log_dir)
    results=sorted(logdir.glob("stage2_*.json"))
    audit.add("Experiments",WARN,"no Stage 2 result files found" if not results else f"{len(results)} Stage 2 result files found")
    for p in results:
        try:
            d=json.loads(p.read_text(encoding="utf-8")); mm=d.get("montgomery_metrics",{}); auc=mm.get("auc"); spec=mm.get("specificity")
            if auc is None: audit.add("Result",WARN,f"{p.name}: no Montgomery AUC")
            elif auc < .70 or (spec is not None and spec < .50): audit.add("Result",WARN,f"{p.name}: Montgomery AUC={auc:.4f}, specificity={spec:.4f}")
            else: audit.add("Result",PASS,f"{p.name}: Montgomery AUC={auc:.4f}")
        except Exception as e: audit.add("Result",FAIL,f"{p.name} unreadable: {e}")

    print("\n[REPRODUCIBILITY / NON-IID]")
    audit.add("Config",PASS,f"hospitals={config.data.num_hospitals}, split={config.data.split_strategy}, alpha={config.data.split_alpha}, seed={config.finetuning.seed}")
    if config.data.split_strategy=="non_iid":
        audit.add("Non-IID",WARN,"NIH is unlabeled in Stage 1; current Dirichlet split represents quantity/statistical skew, not confirmed pathology skew")
    split_files=list(Path(config.data.processed_dir).glob("hospital_*/indices.npy"))
    if split_files: audit.add("Split cache",WARN,"cached indices exist without stored alpha/strategy/seed metadata")
    else: audit.add("Split cache",PASS,"no cached hospital split")

    c=audit.summary()
    out=logdir/"research_audit_report.json"; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"config":args.config,"summary":c,"checks":[{"area":a,"status":s,"message":m} for a,s,m in audit.rows]},indent=2),encoding="utf-8")
    print(f"\nReport saved -> {out}")

if __name__=="__main__": main()
