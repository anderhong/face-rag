# Face RAG API

Face Search in Group Photos using FaceNet + MTCNN + FastAPI.

## Features

- 👤 Face Detection (MTCNN)
- 🔍 Face Recognition (FaceNet)
- 🖼️ Group Photo Search
- ⚙️ Adjustable Threshold
- 🚀 FastAPI + Swagger UI

## Tech Stack

- **MTCNN** - Face Detection
- **FaceNet** (InceptionResnetV1) - Face Recognition
- **FastAPI** - Web Framework
- **Uvicorn** - ASGI Server
- **PyTorch** - Deep Learning

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/face-rag.git
cd face-rag

# Install dependencies
uv sync

# Build face index (first time only)
uv run python build_face_index.py

# Run API
uv run uvicorn main:app --reload --port 8001