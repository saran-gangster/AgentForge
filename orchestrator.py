import os
import sys
import json
import uuid
import time
import base64
import threading
import traceback
import io
import contextlib
import re
from typing import Optional, Any, List, Dict
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel
from filelock import FileLock

import pandas as pd

APP_NAME = "AgentForge Orchestrator (Phase 1+)"
LOG_DIR = os.getenv("LOG_DIR", "./logs")
STATE_DIR = os.getenv("STATE_DIR", "./state")
DATA_DIR = os.getenv("DATA_DIR", "./data")
TOOLS_PATH = os.getenv("TOOLS_PATH", os.path.join(STATE_DIR, "tools.json"))
DATASETS_PATH = os.getenv("DATASETS_PATH", os.path.join(STATE_DIR, "datasets.json"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

LORAS = ["base", "financial_analyst", "marketing_analyst"]
JOBS: Dict[str, Dict[str, Any]] = {}

PLACEHOLDER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)

# =========================
# Pydantic Schemas
# =========================
class AnalyzeRequest(BaseModel):
    prompt: str = ""
    dataset_id: str
    lora_name: Optional[str] = None
    use_golden: bool = False


class Artifact(BaseModel):
    type: str  # "image" | "table" | "json" | "text"
    name: str
    b64: Optional[str] = None
    data: Optional[Any] = None


class Plan(BaseModel):
    reasoning: str
    python_code: str
    expected_artifacts: List[str] = []


class LintResult(BaseModel):
    ok: bool
    message: str
    violations: List[str] = []


class ExecutionResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    artifacts: List[Artifact] = []


class AnalyzeResponse(BaseModel):
    plan: Optional[Plan] = None
    lint: LintResult
    execution: ExecutionResult
    errors: Optional[Dict[str, Any]] = None


class TrainRequest(BaseModel):
    specialist_name: str
    dataset_id: str
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"


class TrainStatus(BaseModel):
    status: str
    logs: List[str] = []


# =========================
# Dataset Registry
# =========================
def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_{2,}", "_", s).strip("_")


def _ensure_datasets_file():
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(os.path.dirname(DATASETS_PATH), exist_ok=True)
        with open(DATASETS_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def load_datasets_json() -> Dict[str, Dict[str, str]]:
    _ensure_datasets_file()
    with open(DATASETS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data


def save_datasets_json(dsets: Dict[str, Dict[str, str]]) -> None:
    lock_path = DATASETS_PATH + ".lock"
    with FileLock(lock_path, timeout=5):
        with open(DATASETS_PATH, "w", encoding="utf-8") as f:
            json.dump(dsets, f, indent=2)


def ensure_sample_datasets() -> None:
    dsets = load_datasets_json()

    # sales_q3: messy columns
    sales_id = "sales_q3"
    sales_name = "Sales Q3.jsonl"
    sales_path = os.path.join(DATA_DIR, sales_name)
    if not os.path.exists(sales_path):
        rows = []
        for wk in range(1, 9):
            rev = 1000 + wk * 37
            cogs = 600 + wk * 21
            sgna = 200 + wk * 9
            rows.append({"wk": wk, "rev": rev, "cogs": cogs, "sgna": sgna})
        with open(sales_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    dsets[sales_id] = {"name": sales_name, "path": sales_path}

    # ad_spend_q3: clean columns
    ad_id = "ad_spend_q3"
    ad_name = "Ad Spend Q3.jsonl"
    ad_path = os.path.join(DATA_DIR, ad_name)
    if not os.path.exists(ad_path):
        rows = []
        for wk in range(1, 9):
            spend = 500 + wk * 10
            impressions = 10000 + wk * 250
            clicks = 300 + wk * 5
            rows.append({"week": wk, "spend": spend, "impressions": impressions, "clicks": clicks})
        with open(ad_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    dsets[ad_id] = {"name": ad_name, "path": ad_path}

    save_datasets_json(dsets)


def list_datasets_public() -> List[Dict[str, str]]:
    dsets = load_datasets_json()
    return [{"id": did, "name": meta.get("name", "")} for did, meta in dsets.items()]


def df_from_dataset(dataset_id: str) -> pd.DataFrame:
    dsets = load_datasets_json()
    meta = dsets.get(dataset_id)
    if not meta:
        raise FileNotFoundError(f"Unknown dataset_id={dataset_id}")
    path = meta["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json"):
        try:
            return pd.read_json(path)
        except Exception:
            # best-effort, assume jsonl
            return pd.read_json(path, lines=True)
    else:
        return pd.read_json(path, lines=True)


# =========================
# Tool Registry
# =========================
def _default_tools_seed() -> List[Dict[str, str]]:
    return [
        {
            "name": "moving_average",
            "desc": "Compute simple rolling mean",
            "code": (
                "import pandas as pd\n"
                "def moving_average(series, window=3):\n"
                "    return pd.Series(series).rolling(window).mean().tolist()\n"
            ),
        },
        {
            "name": "zscore",
            "desc": "Compute z-score for a series",
            "code": (
                "import math\n"
                "def zscore(values):\n"
                "    vals = list(values)\n"
                "    n = len(vals)\n"
                "    if n == 0:\n"
                "        return []\n"
                "    mean = sum(vals)/n\n"
                "    var = sum((x-mean)**2 for x in vals)/n\n"
                "    sd = math.sqrt(var)\n"
                "    return [0.0 if sd==0 else (x-mean)/sd for x in vals]\n"
            ),
        },
    ]


def _ensure_tools_file():
    if not os.path.exists(TOOLS_PATH):
        os.makedirs(os.path.dirname(TOOLS_PATH), exist_ok=True)
        with open(TOOLS_PATH, "w", encoding="utf-8") as f:
            json.dump(_default_tools_seed(), f, indent=2)


def load_tools_json() -> List[Dict[str, str]]:
    _ensure_tools_file()
    with open(TOOLS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            return []
        return data


def save_tools_json(tools: List[Dict[str, str]]) -> None:
    lock_path = TOOLS_PATH + ".lock"
    with FileLock(lock_path, timeout=5):
        with open(TOOLS_PATH, "w", encoding="utf-8") as f:
            json.dump(tools, f, indent=2)


def list_tools_public() -> List[Dict[str, str]]:
    tools = load_tools_json()
    return [{"name": t.get("name", ""), "desc": t.get("desc", "")} for t in tools]


# =========================
# AST Linter
# =========================
import ast


class CodeLinter:
    def __init__(self):
        self.allowed_imports = {
            "pandas",
            "numpy",
            "math",
            "json",
            "statistics",
            "matplotlib",
            "matplotlib.pyplot",
            "seaborn",
        }
        self.banned_calls = {"eval", "exec", "compile", "__import__", "open", "input"}
        self.banned_dotted = {
            "os.system",
            "os.popen",
            "subprocess.run",
            "subprocess.Popen",
            "socket.socket",
            "pathlib.Path",
            "shutil.rmtree",
            "shutil.copyfile",
            "requests.get",
            "requests.post",
            "urllib.request",
        }

    def _get_func_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Call):
            return self._get_func_name(node.func)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return ""

    def lint(self, code: str, require_run_signature: bool = True) -> LintResult:
        violations: List[str] = []
        try:
            tree = ast.parse(code)
        except Exception as e:
            return LintResult(ok=False, message=f"syntax_error: {e}", violations=[str(e)])

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod not in self.allowed_imports:
                        violations.append(f"illegal_import: {mod}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod not in self.allowed_imports:
                    violations.append(f"illegal_import_from: {mod}")
            elif isinstance(node, ast.Call):
                fname = self._get_func_name(node)
                if fname in self.banned_calls:
                    violations.append(f"banned_call: {fname}")
                if fname in self.banned_dotted:
                    violations.append(f"banned_call: {fname}")

        if require_run_signature:
            run_ok = False
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "run":
                    arg_names = [a.arg for a in node.args.args]
                    if arg_names[:2] == ["df", "tools"]:
                        run_ok = True
                    else:
                        violations.append("invalid_run_signature: expected run(df, tools)")
            if not run_ok:
                violations.append("missing_run: define def run(df, tools): ...")

        ok = len(violations) == 0
        return LintResult(ok=ok, message="ok" if ok else "lint_failed", violations=violations)


LINTER = CodeLinter()

# =========================
# Sandbox
# =========================
from multiprocessing import Process, Queue


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    artifacts: List[Dict[str, Any]]
    error: Optional[str] = None


def _sandbox_worker(code: str, df: pd.DataFrame, tools_code: List[Dict[str, str]], q: Queue):
    try:
        import resource  # type: ignore
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    except Exception:
        pass

    # Headless plotting
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa
    except Exception:
        plt = None  # type: ignore

    def build_tools_runtime() -> Dict[str, Any]:
        runtime: Dict[str, Any] = {}
        for t in tools_code:
            name = t.get("name", "")
            code_s = t.get("code", "")
            lr = LINTER.lint(code_s, require_run_signature=False)
            if not lr.ok:
                continue
            safe_builtins = {
                "__import__": __import__,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "map": map,
                "filter": filter,
                "list": list,
                "dict": dict,
                "set": set,
                "sorted": sorted,
                "any": any,
                "all": all,
                "zip": zip,
                "round": round,
                "print": print,
            }
            g: Dict[str, Any] = {
                "__builtins__": safe_builtins,
                "pd": pd,
                "math": __import__("math"),
                "json": __import__("json"),
            }
            l: Dict[str, Any] = {}
            try:
                exec(compile(code_s, "<tool>", "exec"), g, l)
            except Exception:
                continue
            if name and name in l and callable(l[name]):
                runtime[name] = l[name]
            else:
                for k, v in l.items():
                    if callable(v):
                        runtime[k] = v
        return runtime

    tools = build_tools_runtime()

    safe_builtins = {
        "__import__": __import__,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "map": map,
        "filter": filter,
        "list": list,
        "dict": dict,
        "set": set,
        "sorted": sorted,
        "any": any,
        "all": all,
        "zip": zip,
        "round": round,
        "print": print,
    }
    g: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "math": __import__("math"),
        "json": __import__("json"),
    }
    l: Dict[str, Any] = {}

    def capture_matplotlib_artifacts() -> List[Dict[str, Any]]:
        arts: List[Dict[str, Any]] = []
        try:
            import matplotlib.pyplot as plt  # type: ignore
            figs = plt.get_fignums()
            for i, num in enumerate(figs, start=1):
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
                arts.append({"type": "image", "name": f"figure_{i}", "b64": b64})
            if figs:
                plt.close("all")
        except Exception:
            pass
        return arts

    out = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(code, "<plan>", "exec"), g, l)
            if "run" not in l or not callable(l["run"]):
                q.put(SandboxResult(ok=False, stdout=out.getvalue(), stderr=err.getvalue(), artifacts=[], error="no_run").__dict__)
                return
            result = l["run"](df, tools)
            stdout_s = out.getvalue()
            stderr_s = err.getvalue()
            artifacts: List[Dict[str, Any]] = []
            if isinstance(result, dict):
                if isinstance(result.get("stdout"), str):
                    stdout_s += (("\n" if stdout_s else "") + result.get("stdout"))
                if isinstance(result.get("artifacts"), list):
                    artifacts = result.get("artifacts")
            # auto-capture figures
            artifacts += capture_matplotlib_artifacts()
            q.put(SandboxResult(ok=True, stdout=stdout_s, stderr=stderr_s, artifacts=artifacts).__dict__)
    except Exception as e:
        tb = traceback.format_exc()
        q.put(SandboxResult(ok=False, stdout=out.getvalue(), stderr=err.getvalue() + "\n" + tb, artifacts=[], error=str(e)).__dict__)


def run_in_sandbox(code: str, df: pd.DataFrame, tools: List[Dict[str, str]], timeout_s: int = 15) -> SandboxResult:
    q: Queue = Queue()
    p = Process(target=_sandbox_worker, args=(code, df, tools, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        return SandboxResult(ok=False, stdout="", stderr="timeout", artifacts=[], error="timeout")
    if not q.empty():
        data = q.get()
        return SandboxResult(**data)
    return SandboxResult(ok=False, stdout="", stderr="no_output", artifacts=[], error="no_output")


# =========================
# Planner + Golden
# =========================
def golden_plan_sales_q3() -> Plan:
    code = (
        "import pandas as pd\n"
        "def run(df, tools):\n"
        "    df = df.copy()\n"
        "    rename_map = {}\n"
        "    if 'rev' in df.columns: rename_map['rev'] = 'revenue'\n"
        "    if 'cogs' in df.columns: rename_map['cogs'] = 'cogs'\n"
        "    if 'sgna' in df.columns: rename_map['sgna'] = 'sgna'\n"
        "    if 'wk' in df.columns: rename_map['wk'] = 'fiscal_week'\n"
        "    df = df.rename(columns=rename_map)\n"
        "    df['profit'] = df['revenue'] - df['cogs'] - df['sgna']\n"
        "    df['profit_margin'] = (df['profit'] / df['revenue']).replace([float('inf'), float('-inf')], 0).fillna(0)\n"
        "    agg = df.groupby('fiscal_week', dropna=False).agg(revenue=('revenue','sum'), profit=('profit','sum')).reset_index()\n"
        "    artifacts = [{ 'type':'table', 'name':'profit_by_week', 'data': agg.to_dict(orient='records') }]\n"
        "    return {'stdout': f\"rows={len(df)}, weeks={len(agg)}\", 'artifacts': artifacts}\n"
    )
    return Plan(
        reasoning="Normalize schema (rev->revenue, wk->fiscal_week), derive profit and margin, aggregate by week.",
        python_code=code,
        expected_artifacts=["table:profit_by_week"],
    )


def planner_generate(req: AnalyzeRequest) -> Plan:
    lora = (req.lora_name or "base").strip().lower()
    if req.dataset_id == "sales_q3":
        if lora == "financial_analyst":
            return golden_plan_sales_q3()
        else:
            code = (
                "import os\n"  # banned
                "import pandas as pd\n"
                "def run(df, tools):\n"
                "    df['profit'] = df['revenue'] - df['cogs'] - df['sgna']\n"
                "    agg = df.groupby('fiscal_week').agg(revenue=('revenue','sum'), profit=('profit','sum')).reset_index()\n"
                "    return {'stdout': 'done', 'artifacts': [{'type':'table','name':'profit_by_week','data': agg.to_dict(orient='records')}]} \n"
            )
            return Plan(
                reasoning="Generalist assumes clean schema and imports an unsafe module.",
                python_code=code,
                expected_artifacts=["table:profit_by_week"],
            )
    else:
        code = (
            "import pandas as pd\n"
            "def run(df, tools):\n"
            "    df = df.copy()\n"
            "    if 'impressions' in df.columns and 'clicks' in df.columns:\n"
            "        df['ctr'] = (df['clicks'] / df['impressions']).fillna(0)\n"
            "    if 'spend' in df.columns and 'clicks' in df.columns:\n"
            "        df['cpc'] = (df['spend'] / df['clicks']).replace([float('inf'), float('-inf')], 0).fillna(0)\n"
            "    artifacts = [{'type':'table','name':'ad_metrics','data': df.to_dict(orient='records')}]\n"
            "    return {'stdout': f\"rows={len(df)}\", 'artifacts': artifacts}\n"
        )
        return Plan(
            reasoning="Compute CTR and CPC; return annotated table.",
            python_code=code,
            expected_artifacts=["table:ad_metrics"],
        )


def golden_for_dataset(dataset_id: str) -> Optional[Plan]:
    if dataset_id == "sales_q3":
        return golden_plan_sales_q3()
    return None


# =========================
# Training (stub)
# =========================
def _run_job(job_id: str, specialist_name: str, dataset_id: str, base_model: str):
    logs: List[str] = []

    def log(msg: str):
        logs.append(msg)
        JOBS[job_id]["logs"] = logs[:]

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

        if specialist_name not in LORAS:
            LORAS.append(specialist_name)

        JOBS[job_id]["status"] = "succeeded"
        log("Training completed successfully.")
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        log(f"Error: {e}")


# =========================
# FastAPI app
# =========================
app = FastAPI(title=APP_NAME)


@app.on_event("startup")
def _startup():
    ensure_sample_datasets()
    _ensure_tools_file()


@app.get("/loras")
def list_loras():
    return {"loras": LORAS}


@app.get("/datasets")
def list_datasets():
    return {"datasets": list_datasets_public()}


@app.post("/datasets/upload")
async def datasets_upload(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    filename = file.filename or "dataset"
    base, ext = os.path.splitext(filename)
    if ext.lower() not in {".csv", ".jsonl", ".json"}:
        raise HTTPException(status_code=400, detail="unsupported_file_type")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty_file")

    # create unique id and path
    display_name = name.strip() if name else filename
    did_base = _slugify(os.path.splitext(display_name)[0]) or "dataset"
    dsets = load_datasets_json()
    did = did_base
    i = 1
    while did in dsets:
        i += 1
        did = f"{did_base}_{i}"

    save_name = f"{did}{ext.lower()}"
    path = os.path.join(DATA_DIR, save_name)
    with open(path, "wb") as f:
        f.write(contents)

    dsets[did] = {"name": display_name, "path": path}
    save_datasets_json(dsets)

    return {"id": did, "name": display_name}


@app.get("/datasets/{dataset_id}/preview")
def datasets_preview(dataset_id: str, rows: int = Query(20, ge=1, le=200)):
    try:
        df = df_from_dataset(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"dataset_error: {e}")
    head = df.head(rows)
    cols = list(map(str, head.columns.tolist()))
    data = head.to_dict(orient="records")
    return {"columns": cols, "rows": data, "count": len(data)}


@app.get("/tools")
def list_tools():
    return {"tools": list_tools_public()}


@app.post("/tools/register")
def register_tool(payload: Dict[str, Any]):
    name = (payload.get("name") or "").strip()
    desc = (payload.get("desc") or "").strip()
    code = payload.get("code") or ""

    if not name:
        raise HTTPException(status_code=400, detail="name_required")

    lr = LINTER.lint(code, require_run_signature=False)
    if not lr.ok:
        raise HTTPException(status_code=400, detail={"message": "lint_failed", "violations": lr.violations})

    tools = load_tools_json()
    existing = next((t for t in tools if t.get("name") == name), None)
    if existing:
        existing["desc"] = desc or existing.get("desc", "")
        existing["code"] = code or existing.get("code", "")
    else:
        tools.append({"name": name, "desc": desc, "code": code})
    save_tools_json(tools)
    return {"message": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    plan = planner_generate(req)

    lint_res = LINTER.lint(plan.python_code, require_run_signature=True)
    if not lint_res.ok:
        if req.use_golden:
            golden = golden_for_dataset(req.dataset_id)
            if golden:
                plan = golden
                lint_res = LINTER.lint(plan.python_code, require_run_signature=True)
        if not lint_res.ok:
            return AnalyzeResponse(
                plan=plan,
                lint=lint_res,
                execution=ExecutionResult(stdout="", stderr="", artifacts=[]),
                errors=None,
            )

    try:
        df = df_from_dataset(req.dataset_id)
    except Exception as e:
        return AnalyzeResponse(
            plan=plan,
            lint=lint_res,
            execution=ExecutionResult(stdout="", stderr="", artifacts=[]),
            errors={"message": f"dataset_error: {e}"},
        )

    tools = load_tools_json()
    sb_res = run_in_sandbox(plan.python_code, df, tools, timeout_s=15)

    if (not sb_res.ok) and req.use_golden:
        golden = golden_for_dataset(req.dataset_id)
        if golden:
            plan = golden
            lint_res = LINTER.lint(plan.python_code, require_run_signature=True)
            if lint_res.ok:
                sb_res = run_in_sandbox(plan.python_code, df, tools, timeout_s=15)

    exec_res = ExecutionResult(
        stdout=sb_res.stdout,
        stderr=sb_res.stderr,
        artifacts=[Artifact(**a) for a in sb_res.artifacts] if sb_res.artifacts else [],
    )
    return AnalyzeResponse(
        plan=plan,
        lint=lint_res,
        execution=exec_res,
        errors=None if sb_res.ok else {"message": sb_res.error or "execution_failed"},
    )


@app.post("/train")
def start_train(req: TrainRequest):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "logs": []}
    t = threading.Thread(target=_run_job, args=(job_id, req.specialist_name, req.dataset_id, req.base_model), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/train/{job_id}/status")
def train_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="not_found")
    return TrainStatus(status=job.get("status", "unknown"), logs=job.get("logs", []))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    print(f"{APP_NAME} running on http://127.0.0.1:{port}")
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=port, reload=True)