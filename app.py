import os
import time
import html
from flask import Flask, render_template, request, Response

from orchestrator_client import Orchestrator

ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8001")
GEN_URL = os.getenv("GENERATOR_URL", "")
LOG_DIR = os.getenv("LOG_DIR", "./logs")

app = Flask(__name__)
orch = Orchestrator(ORCH_URL, GEN_URL)


def _safe_call(fn, fallback, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] {getattr(fn, '__name__', str(fn))}: {e}")
        return fallback


@app.route("/")
def index():
    loras = _safe_call(orch.list_loras, ["base"])
    datasets = _safe_call(orch.list_datasets, [])
    tools = _safe_call(orch.list_tools, [])
    return render_template("index.html", loras=loras, datasets=datasets, tools=tools)


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = {
        "prompt": request.form.get("prompt", ""),
        "dataset_id": request.form.get("dataset_id"),
        "lora_name": request.form.get("lora_name") or None,
        "use_golden": request.form.get("use_golden") == "on",
    }
    try:
        resp = orch.analyze(payload)
    except Exception as e:
        resp = {
            "plan": None,
            "lint": {"ok": False, "message": "orchestrator_error", "violations": []},
            "execution": {"stdout": "", "stderr": "", "artifacts": []},
            "errors": {"message": str(e)},
        }
    return render_template("partials/results.html", resp=resp)


@app.route("/train", methods=["POST"])
def train():
    specialist_name = request.form.get("specialist_name")
    dataset_id = request.form.get("dataset_id")
    base_model = request.form.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct")
    out = _safe_call(orch.start_training, {"job_id": "error"}, specialist_name, dataset_id, base_model)
    job_id = out.get("job_id")
    return render_template("partials/training_status.html", job_id=job_id, status="queued")


@app.route("/train/status/<job_id>")
def train_status(job_id):
    status = _safe_call(orch.training_status, {"status": "unknown", "logs": []}, job_id)
    return render_template(
        "partials/training_status.html",
        job_id=job_id,
        status=status.get("status"),
        logs=status.get("logs", []),
    )


@app.route("/events")
def sse_events():
    job_id = request.args.get("job_id")
    if not job_id:
        return ("Missing job_id", 400)

    log_path = os.path.join(LOG_DIR, f"{job_id}.log")

    def ev(event: str, data: str) -> str:
        return f"event: {event}\ndata: {data}\n\n"

    def gen():
        last_status = None
        last_len = 0
        f = None
        pos = 0

        yield "retry: 1000\n\n"
        yield ": keep-alive\n\n"

        while True:
            if f is None and os.path.exists(log_path):
                try:
                    f = open(log_path, "r", encoding="utf-8", errors="replace")
                    f.seek(0, os.SEEK_END)
                    pos = f.tell()
                except Exception:
                    f = None

            try:
                status_resp = orch.training_status(job_id)
            except Exception:
                yield ev("status", f'<span id="status-value">{html.escape("error")}</span>')
                yield ev("done", "1")
                break

            status = status_resp.get("status")
            logs = status_resp.get("logs", [])

            if status and status != last_status:
                last_status = status
                yield ev("status", f'<span id="status-value">{html.escape(status)}</span>')

            if f is not None:
                try:
                    f.seek(pos)
                    new_lines = f.readlines()
                    if new_lines:
                        for line in new_lines:
                            safe = html.escape(line.rstrip("\n"))
                            yield ev("log", safe + "\n")
                        pos = f.tell()
                except Exception:
                    f = None
            else:
                if last_len < len(logs):
                    for line in logs[last_len:]:
                        safe = html.escape(str(line))
                        yield ev("log", safe + "\n")
                    last_len = len(logs)

            if status in {"succeeded", "failed"}:
                yield ev("done", "1")
                break

            time.sleep(0.5)

        try:
            if f:
                f.close()
        except Exception:
            pass

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(gen(), mimetype="text/event-stream", headers=headers)


@app.route("/tools/add", methods=["POST"])
def tools_add():
    name = request.form.get("name")
    code = request.form.get("code")
    desc = request.form.get("desc", "")
    _safe_call(orch.register_tool, {"message": "error"}, name, code, desc)
    tools = _safe_call(orch.list_tools, [])
    return render_template("partials/tools.html", tools=tools, message="Updated")


@app.route("/datasets/refresh")
def datasets_refresh():
    datasets = _safe_call(orch.list_datasets, [])
    return render_template("partials/datasets.html", datasets=datasets)


@app.route("/datasets/upload", methods=["POST"])
def datasets_upload():
    name = request.form.get("name")
    f = request.files.get("file")
    if not f:
        datasets = _safe_call(orch.list_datasets, [])
        return render_template("partials/datasets.html", datasets=datasets, message="No file provided")
    try:
        uploaded = orch.upload_dataset(f.read(), f.filename or "dataset", name)
        msg = f"Uploaded: {uploaded.get('name')} (id={uploaded.get('id')})"
    except Exception as e:
        msg = f"Upload error: {e}"
    datasets = _safe_call(orch.list_datasets, [])
    return render_template("partials/datasets.html", datasets=datasets, message=msg)


@app.route("/datasets/preview/<dataset_id>")
def datasets_preview(dataset_id):
    data = _safe_call(orch.dataset_preview, {"columns": [], "rows": [], "count": 0}, dataset_id, 20)
    return render_template("partials/dataset_preview.html", preview=data, dataset_id=dataset_id)


@app.route("/loras/refresh")
def loras_refresh():
    loras = _safe_call(orch.list_loras, ["base"])
    return render_template("partials/loras.html", loras=loras)


@app.route("/tools/refresh")
def tools_refresh():
    tools = _safe_call(orch.list_tools, [])
    return render_template("partials/tools.html", tools=tools)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)