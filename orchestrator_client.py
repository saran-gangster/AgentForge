import requests
from typing import Optional


class Orchestrator:
    def __init__(self, orchestrator_url: str, generator_url: str | None = None):
        self.base = orchestrator_url.rstrip("/")
        self.gen = (generator_url or "").rstrip("/")

    def analyze(self, payload: dict) -> dict:
        r = requests.post(f"{self.base}/analyze", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def list_tools(self) -> list[dict]:
        r = requests.get(f"{self.base}/tools", timeout=10)
        r.raise_for_status()
        return r.json().get("tools", [])

    def register_tool(self, name: str, code: str, desc: str = "") -> dict:
        r = requests.post(
            f"{self.base}/tools/register",
            json={"name": name, "code": code, "desc": desc},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def list_loras(self) -> list[str]:
        r = requests.get(f"{self.base}/loras", timeout=10)
        r.raise_for_status()
        return r.json().get("loras", [])

    def list_datasets(self) -> list[dict]:
        r = requests.get(f"{self.base}/datasets", timeout=10)
        r.raise_for_status()
        return r.json().get("datasets", [])

    def upload_dataset(self, file_bytes: bytes, filename: str, name: Optional[str] = None) -> dict:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        data = {}
        if name:
            data["name"] = name
        r = requests.post(f"{self.base}/datasets/upload", files=files, data=data, timeout=60)
        r.raise_for_status()
        return r.json()

    def dataset_preview(self, dataset_id: str, rows: int = 20) -> dict:
        r = requests.get(f"{self.base}/datasets/{dataset_id}/preview", params={"rows": rows}, timeout=15)
        r.raise_for_status()
        return r.json()

    def start_training(self, specialist_name: str, dataset_id: str, base_model: str) -> dict:
        r = requests.post(
            f"{self.base}/train",
            json={
                "specialist_name": specialist_name,
                "dataset_id": dataset_id,
                "base_model": base_model,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def training_status(self, job_id: str) -> dict:
        r = requests.get(f"{self.base}/train/{job_id}/status", timeout=10)
        r.raise_for_status()
        return r.json()