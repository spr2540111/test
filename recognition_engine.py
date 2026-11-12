import cv2
import numpy as np

from insightface.app import FaceAnalysis
from src.utils import cosine_similarity


THRESHOLD = 0.75

# -------------------------
# InsightFace初期化
# -------------------------
app = FaceAnalysis()
app.prepare(ctx_id=0)

# -------------------------
# DBロード
# -------------------------
embeddings = np.load(
    "embeddings/face_embeddings.npy",
    allow_pickle=True
)

names = np.load(
    "embeddings/face_names.npy",
    allow_pickle=True
)


def recognize_face(image):

    faces = app.get(image)

    if len(faces) == 0:

        return {
            "success": False,
            "message": "face_not_found"
        }

    face = faces[0]

    embedding = face.embedding

    best_score = -1
    best_name = "Unknown"

    for db_embedding, db_name in zip(
        embeddings,
        names
    ):

        score = cosine_similarity(
            embedding,
            db_embedding
        )

        if score > best_score:

            best_score = score
            best_name = db_name

    if best_score < THRESHOLD:

        return {
            "success": False,
            "message": "unknown"
        }

    return {
        "success": True,
        "name": str(best_name),
        "score": float(best_score)
    }