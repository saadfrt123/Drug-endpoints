# FastAPI AI Endpoints – Minimal Deployment Package

This folder contains only the files required to deploy the FastAPI AI endpoints (classification + cascade prediction) on an EC2 instance or any Linux host.

## 📁 Contents
| File | Purpose |
|------|---------|
| `main.py` | FastAPI application with all endpoints |
| `config.py` | Loads environment variables and exposes configuration object |
| `requirements.txt` | Python dependencies |
| `env.example` | Template for `.env` file |
| `fastapi-drug-target.service` | Systemd unit file (optional, for auto-start) |

## 🚀 Quick Deployment Steps

1. **Copy files to server**
   ```bash
   scp -i your-key.pem -r deployment_minimal/ ubuntu@your-ec2:/home/ubuntu/fastapi-ai/
   ```

2. **Create virtual environment & install deps**
   ```bash
   cd /home/ubuntu/fastapi-ai
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp env.example .env
   nano .env   # Fill in NEO4J + GEMINI credentials, host/port, API key
   ```

4. **Run locally for sanity check**
   ```bash
   source venv/bin/activate
   uvicorn main:app --host 127.0.0.1 --port 8001 --reload
   curl http://127.0.0.1:8001/health
   ```

5. **(Optional) Install as a service**
   ```bash
   sudo cp fastapi-drug-target.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable fastapi-drug-target
   sudo systemctl start fastapi-drug-target
   sudo systemctl status fastapi-drug-target
   ```

> **Reminder:** The application bundles `mechanism_classifier.py` and `cascade_predictor.py` in this directory, so you can deploy this folder standalone. If you co-locate it inside the larger Streamlit project, the import path will still work thanks to the fallback search order.

---
**Need more guidance?** See the full deployment docs in `deployment/README.md`, but this folder stays lean for quick installations.
