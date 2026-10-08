from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os

# ============ 1. Load models ============
print("📥 Loading Face Detection & Recognition models...")
mtcnn = MTCNN(keep_all=True, device="cpu")
face_model = InceptionResnetV1(pretrained="vggface2").eval()
print("✅ Models loaded\n")


def embed_faces(image_path):
    """Detect, crop, and embed all faces."""
    image = Image.open(image_path).convert("RGB")
    faces = mtcnn(image)
    
    if faces is None:
        return None
    
    with torch.no_grad():
        embeddings = face_model(faces)
    
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


# ============ 2. Load the database ============
if not os.path.exists("face_embeddings.pt"):
    print("❌ face_embeddings.pt not found. Please run build_face_index.py first.")
    exit(1)

data = torch.load("face_embeddings.pt", weights_only=False)
db_embeddings = data["embeddings"]
db_paths = data["paths"]
db_face_indices = data["face_indices"]

print(f"📦 Loaded {len(db_embeddings)} face embeddings from {len(set(db_paths))} unique images\n")


# ============ 3. Face Search ============
def search_group_photo(group_photo_path: str, threshold: float = 0.7):
    """
    Identify the people in a group photo who appear in the database.
    
    Args:
        group_photo_path: Path to the group photo.
        threshold: Similarity threshold (FaceNet typically treats values of 0.7 or higher as the same person).
    """
    print(f"🖼️ Query Group Photo: {group_photo_path}")
    print(f"   Threshold: {threshold}")
    print(f"{'='*70}")
    
    # Step 1: Detect all faces in the group photo.
    query_embeddings = embed_faces(group_photo_path)
    
    if query_embeddings is None:
        print("   ❌ No faces detected in the group photo.")
        return []
    
    print(f"   🔍 Detected {len(query_embeddings)} face(s) in group photo\n")
    
    # Step 2: Find the most similar database face for each query face.
    matches = []
    
    for i, query_emb in enumerate(query_embeddings):
        # Calculate similarity against all database faces.
        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), db_embeddings)
        
        # Find the highest similarity score.
        best_idx = torch.argmax(similarities).item()
        best_sim = similarities[best_idx].item()
        best_path = db_paths[best_idx]
        
        print(f"   👤 Face #{i+1} in group photo:")
        print(f"      Best Match: {best_path}")
        print(f"      Similarity: {best_sim:.3f}")
        
        if best_sim >= threshold:
            print(f"      ✅ MATCH! (Similarity >= {threshold})")
            matches.append({
                "query_face_index": i,
                "matched_path": best_path,
                "similarity": best_sim
            })
        else:
            print(f"      ❌ No match (Similarity < {threshold})")
        print()
    
    # Step 3: Summarize the results.
    print(f"{'='*70}")
    if matches:
        print(f"🎯 Found {len(matches)} match(es):")
        for m in matches:
            print(f"   - Face #{m['query_face_index']+1} → {m['matched_path']} ({m['similarity']:.3f})")
    else:
        print("   ❌ No matches found.")
    print()
    
    return matches


# ============ 4. Interactive Loop ============
if __name__ == "__main__":
    print("="*50)
    print("👤 Face Search in Group Photos")
    print("="*50)
    print("💡 用法:")
    print("   - 輸入群體照路徑: ./input_images/query1.jpg")
    print("   - 輸入 'bye' 或 'exit' 結束")
    print("="*50)
    
    while True:
        user_input = input("\n🔍 Group Photo Path: ").strip()
        
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("👋 Bye!")
            break
        
        if not user_input:
            continue
        
        if not os.path.exists(user_input):
            print(f"⚠️ Image not found: {user_input}")
            continue
        
        search_group_photo(user_input, threshold=0.7)
