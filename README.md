# Helios Flask UI (Python-First Frontend)

A lightweight, production-minded Flask UI for Helios using:
- Flask + Jinja2 (server-rendered HTML)
- HTMX for partial page updates (no full reloads)
- SSE (auto-connected) for live training logs and status
- Pico.css (CDN) for clean styling

A Mock Orchestrator is included so you can run the full flow locally.

## Requirements

- Python 3.10+ (macOS, Linux, Windows)
- pip

## Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt