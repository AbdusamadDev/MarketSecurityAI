import os
import time
import numpy as np
import faiss
import insightface
import cv2


class FaceRecognizer:
    def __init__(self):
        # Initialize face model
        self.face_model = insightface.app.FaceAnalysis()
        self.face_model.prepare(ctx_id=0)  # GPU mode

        # FAISS index
        self.index = faiss.IndexFlatL2(
            512
        )  # 512 is the embedding size for most deep face models
        self.usernames = []

    def load_face_encodings(self, root_path="../media"):
        all_encodings = []

        # Iterate through each user's directory
        for username in os.listdir(root_path):
            img_path = os.path.join(root_path, username, "main.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self.face_model.get(img)
                if faces and len(faces) > 0:
                    normalized_embedding = faces[0].embedding / np.linalg.norm(
                        faces[0].embedding
                    )
                    all_encodings.append(normalized_embedding)
                    self.usernames.append(username)

        all_encodings = np.array(all_encodings)
        self.index.add(all_encodings)

    def estimate_distance(self, face_width, reference_width, focal_length):
        return (focal_length * reference_width) / face_width

    def recognize(self, img_path, focal_length=50, reference_width=150):
        start_time = time.time()

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get face encoding for the given image
        faces = self.face_model.get(img)
        if not faces or len(faces) == 0:
            return None

        face_width = faces[0].bbox[2] - faces[0].bbox[0]  # x2 - x1
        distance = self.estimate_distance(face_width, reference_width, focal_length)

        if distance > 2.2:
            return {"message": f"Face: {distance}"}

        # Search in FAISS
        normalized_embedding = faces[0].embedding / np.linalg.norm(faces[0].embedding)
        D, I = self.index.search(np.array([normalized_embedding]), 1)
        similarity_percentage = D[0][0] * 100  # Convert to percentage

        elapsed_time = time.time() - start_time

        return {
            "username": self.usernames[I[0][0]],
            "similarity_percentage": similarity_percentage,
            "distance": distance,
            "time_elapsed": elapsed_time,
        }


# Usage
recognizer = FaceRecognizer()
recognizer.load_face_encodings()
result = recognizer.recognize("media/i.jpg")
print(result)
