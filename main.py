from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os
import io

# ============ 1. 建立 FastAPI App ============
app = FastAPI(
    title="Face RAG API",
    description="Face Search in Group Photos using FaceNet",
    version="1.0.0"
)

# ============ 2. 載入 Models ============
print("Loading Face Detection & Recognition models...")
mtcnn = MTCNN(keep_all=True, device="cpu")
face_model = InceptionResnetV1(pretrained="vggface2").eval()
print("Models loaded")

# 載入 Face Embeddings
if not os.path.exists("face_embeddings.pt"):
    raise RuntimeError("face_embeddings.pt not found. Run build_face_index.py first.")

data = torch.load("face_embeddings.pt", weights_only=False)
db_embeddings = data["embeddings"]
db_paths = data["paths"]
db_face_indices = data["face_indices"]

print(f"Loaded {len(db_embeddings)} face embeddings from {len(set(db_paths))} unique images")


# ============ 3. Helper Functions ============
def embed_faces(image: Image.Image):
    """Detect + Crop + Embed 所有人臉"""
    faces = mtcnn(image)
    if faces is None:
        return None
    with torch.no_grad():
        embeddings = face_model(faces)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


# ============ 4. Request Models ============
class FaceSearchResponse(BaseModel):
    filename: str
    faces_detected: int
    threshold: float
    matches_found: int
    matches: list


# ============ 5. API Endpoints ============
@app.get("/")
def root():
    """Health Check"""
    return {
        "status": "ok",
        "message": "Face RAG API",
        "total_faces": len(db_embeddings),
        "total_images": len(set(db_paths))
    }


@app.post("/search/face")
async def search_face(
    file: UploadFile = File(...),
    threshold: float = 0.7
):
    """
    上傳群體照，搵出 Database 入面 Match 嘅人
    
    Args:
        file: 群體照圖片
        threshold: Similarity 門檻（Default 0.7）
    """
    try:
        # 讀取上傳嘅圖
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Detect + Embed
        query_embeddings = embed_faces(image)
        
        if query_embeddings is None:
            return {
                "filename": file.filename,
                "faces_detected": 0,
                "threshold": threshold,
                "matches_found": 0,
                "matches": []
            }
        
        # 對每張 Query Face 搵最佳 Match
        matches = []
        for i, query_emb in enumerate(query_embeddings):
            similarities = torch.cosine_similarity(
                query_emb.unsqueeze(0),
                db_embeddings
            )
            best_idx = torch.argmax(similarities).item()
            best_sim = similarities[best_idx].item()
            
            if best_sim >= threshold:
                matches.append({
                    "query_face_index": i,
                    "matched_path": db_paths[best_idx],
                    "similarity": round(best_sim, 4)
                })
        
        return {
            "filename": file.filename,
            "faces_detected": len(query_embeddings),
            "threshold": threshold,
            "matches_found": len(matches),
            "matches": matches
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faces")
def list_faces():
    """列出所有 Index 咗嘅人臉"""
    return {
        "total_faces": len(db_embeddings),
        "unique_images": len(set(db_paths)),
        "faces": [
            {
                "path": path,
                "face_index": face_idx
            }
            for path, face_idx in zip(db_paths, db_face_indices)
        ]
    }