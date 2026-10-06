from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os

# ============ 1. 載入 Models ============
print("📥 Loading Face Detection & Recognition models...")
mtcnn = MTCNN(keep_all=True, device="cpu")  # keep_all=True 支援多人
face_model = InceptionResnetV1(pretrained="vggface2").eval()
print("✅ Models loaded\n")


def embed_faces(image_path):
    """Detect + Crop + Embed 所有人臉"""
    image = Image.open(image_path).convert("RGB")
    faces = mtcnn(image)  # 回傳 Tensor (N, 3, 160, 160)
    
    if faces is None:
        return None  # 冇 Detect 到臉
    
    with torch.no_grad():
        embeddings = face_model(faces)  # (N, 512)
    
    # Normalize
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


# ============ 2. Indexing Phase ============
face_folder = "./faces"
face_files = sorted([f for f in os.listdir(face_folder) if f.endswith((".jpg", ".jpeg", ".png"))])
face_paths = [os.path.join(face_folder, f) for f in face_files]

print(f"👤 Loading {len(face_paths)} face images...")

all_embeddings = []
all_paths = []
all_face_indices = []

for path in face_paths:
    embeddings = embed_faces(path)
    
    if embeddings is None:
        print(f"   ⚠️ No face detected in {path}, skipping")
        continue
    
    print(f"   ✅ {path} - {len(embeddings)} face(s) detected")
    
    for i in range(len(embeddings)):
        all_embeddings.append(embeddings[i])
        all_paths.append(path)
        all_face_indices.append(i)

if not all_embeddings:
    print("❌ No faces found. Exiting.")
    exit(1)

# Stack 所有 Embeddings
all_embeddings = torch.stack(all_embeddings)

# ============ 3. 儲存 ============
data = {
    "embeddings": all_embeddings,
    "paths": all_paths,
    "face_indices": all_face_indices,
}

torch.save(data, "face_embeddings.pt")
print(f"\n💾 Saved to face_embeddings.pt")
print(f"   Total faces: {len(all_embeddings)}")
print(f"   Embedding shape: {all_embeddings.shape}")