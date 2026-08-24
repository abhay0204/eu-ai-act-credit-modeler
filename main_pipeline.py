import sys
import time
import subprocess
import os

STEPS = [
    {
        "name": "Phase 3.1: Data Ingestion & Cleaning",
        "script": "tools/0_fetch_and_clean_data.py"
    },
    {
        "name": "Phase 3.2: PyTorch Model Training",
        "script": "tools/1_train_baseline.py"
    },
    {
        "name": "Phase 3.3: Fairness Audit & Bias Mitigation",
        "script": "tools/2_audit_and_mitigate.py"
    },
    {
        "name": "Phase 3.4: SHAP Feature Explainability",
        "script": "tools/3_explain_shap.py"
    },
    {
        "name": "Phase 3.5: Model Card Generation",
        "script": "tools/4_generate_model_card.py"
    }
]

def run_step(step_idx, step_info):
    name = step_info["name"]
    script = step_info["script"]
    
    print(f"[INFO] [{step_idx+1}/{len(STEPS)}] Running {name} ({script})...", flush=True)
    start_time = time.time()
    
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[INFO] [{step_idx+1}/{len(STEPS)}] {name} completed in {elapsed:.2f}s.", flush=True)
    else:
        print(f"[ERROR] [{step_idx+1}/{len(STEPS)}] {name} failed with exit code {result.returncode}.", flush=True)
        sys.exit(result.returncode)

def main():
    pipeline_start = time.time()
    print("[INFO] Credit Risk Pipeline Execution Started", flush=True)
    
    for idx, step in enumerate(STEPS):
        run_step(idx, step)
        
    total_elapsed = time.time() - pipeline_start
    print(f"[INFO] Pipeline execution completed successfully in {total_elapsed:.2f}s.", flush=True)

if __name__ == '__main__':
    main()
