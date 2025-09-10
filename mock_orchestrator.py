import os
import threading
import time
import uuid
from flask import Flask, request, jsonify

app = Flask(__name__)

LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)

# In-memory state for demo purposes
TOOLS = [
    {"name": "moving_average", "desc": "Compute simple rolling mean"},
    {"name": "zscore", "desc": "Compute z-score for a series"},
]
LORAS = ["base", "financial_analyst", "marketing_analyst"]
DATASETS = [
    {"id": "sales_q3", "name": "Sales Q3.jsonl"},
    {"id": "ad_spend_q3", "name": "Ad Spend Q3.jsonl"},
]

# job_id -> {"status": str, "logs": [str]}
JOBS: dict[str, dict] = {}

# 1x1 transparent PNG base64 (no data: prefix)
PLACEHOLDER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


@app.get("/loras")
def list_loras():
    return jsonify({"loras": LORAS})


@app.get("/datasets")
def list_datasets():
    return jsonify({"datasets": DATASETS})


@app.get("/tools")
def list_tools():
    return jsonify({"tools": TOOLS})


@app.post("/tools/register")
def register_tool():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    desc = (data.get("desc") or "").strip()
    code = data.get("code") or ""
    if not name:
        return jsonify({"message": "name_required"}), 400
    # De-dup by name
    existing = next((t for t in TOOLS if t["name"] == name), None)
    if existing:
        existing["desc"] = desc or existing.get("desc", "")
    else:
        TOOLS.append({"name": name, "desc": desc})
    return jsonify({"message": "ok"})


@app.post("/analyze")
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt", "")
    dataset_id = data.get("dataset_id", "")
    lora_name = data.get("lora_name") or "base"
    use_golden = bool(data.get("use_golden"))

    plan = {
        "reasoning": f"Step-by-step reasoning for: {prompt}\nUsing dataset: {dataset_id}\nBrain: {lora_name}",
        "python_code": (
            "import pandas as pd\n"
            "def run(df, tools):\n"
            "    # Example analysis placeholder\n"
            "    result = {'summary': 'ok', 'rows': len(df) if hasattr(df,'__len__') else 0}\n"
            "    return {'stdout': str(result), 'artifacts': []}\n"
        ),
        "expected_artifacts": ["chart:demo_chart"],
    }
    lint = {"ok": True, "message": "ok"}
    execution = {
        "stdout": f"Ran analysis with dataset={dataset_id}, lora={lora_name}, golden={use_golden}",
        "stderr": "",
        "artifacts": [
            {"type": "image", "name": "demo_chart", "b64": PLACEHOLDER_PNG_B64}
        ],
    }
    return jsonify({"plan": plan, "lint": lint, "execution": execution, "errors": None})


def _run_job(job_id: str, specialist_name: str, dataset_id: str, base_model: str):
    logs = []
    log_path = os.path.join(LOG_DIR, f"{job_id}.log")
    fp = open(log_path, "a", buffering=1, encoding="utf-8")  # line-buffered, auto-flush

    def log(msg: str):
        logs.append(msg)
        JOBS[job_id]["logs"] = logs[:]  # copy for safety
        try:
            fp.write(msg + "\n")
        except Exception:
            pass

    try:
        JOBS[job_id]["status"] = "queued"
        log(f"Queued training job {job_id} for {specialist_name} using {base_model} on {dataset_id}")
        time.sleep(1.0)

        JOBS[job_id]["status"] = "provisioning"
        log("Provisioning GPU resources...")
        time.sleep(1.2)

        JOBS[job_id]["status"] = "running"
        for i in range(1, 6):
            log(f"[{i}/5] Training epoch {i}...")
            time.sleep(0.8)
        log("Saving LoRA weights to /runpod-volume/loras/" + specialist_name)

        # Pretend new LoRA appears
        if specialist_name not in LORAS:
            LORAS.append(specialist_name)

        JOBS[job_id]["status"] = "succeeded"
        log("Training completed successfully.")
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        log(f"Error: {e}")
    finally:
        try:
            fp.close()
        except Exception:
            pass


@app.post("/train")
def start_train():
    data = request.get_json(force=True, silent=True) or {}
    specialist_name = data.get("specialist_name") or "new_specialist"
    dataset_id = data.get("dataset_id") or "unknown"
    base_model = data.get("base_model") or "meta-llama/Meta-Llama-3-8B-Instruct"

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "logs": []}

    t = threading.Thread(target=_run_job, args=(job_id, specialist_name, dataset_id, base_model), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.get("/train/<job_id>/status")
def train_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "not_found", "logs": []}), 404
    return jsonify({"status": job["status"], "logs": job.get("logs", [])})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    print(f"Mock Orchestrator running on http://127.0.0.1:{port}, logs at {os.path.abspath(LOG_DIR)}")
    app.run(host="0.0.0.0", port=port, debug=True)