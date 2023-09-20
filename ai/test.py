import insightface
import numpy as np
import os
import faiss
import time
import cv2  # Add the OpenCV import


class FaceMatcher:
    def __init__(self, media_path):
        self.media_path = media_path
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=-1)
        self.index, self.names = self._build_faiss_index()

    def _build_faiss_index(self):
        # Prepare the FAISS index
        index = faiss.IndexFlatL2(512)  # 512-dimensional embeddings from insightface
        all_embeddings = []
        names = []

        # Traverse media folder and its subfolders to read all face images
        for person_folder in os.listdir(self.media_path):
            person_folder_path = os.path.join(self.media_path, person_folder)
            for img_name in os.listdir(person_folder_path):
                img_path = os.path.join(person_folder_path, img_name)
                img = cv2.imread(img_path)  # Read the image using OpenCV
                face = self.model.get(img)  # Pass the image array, not the path
                if face:
                    embedding = face[0].embedding
                    all_embeddings.append(embedding)
                    names.append(person_folder)

        # Add embeddings to FAISS index
        index.add(np.array(all_embeddings))
        return index, names

    def match_face(self, img_path):
        start_time = time.time()

        img = cv2.imread(img_path) 
        face = self.model.get(img) 
        if not face:
            return None, None, None

        embedding = face[0].embedding
        D, I = self.index.search(np.array([embedding]), 1)

        end_time = time.time()
        processing_time = end_time - start_time

        similarity = D[0][0]
        name = self.names[I[0][0]]

        return processing_time, similarity, name


if __name__ == "__main__":
    media_folder_path = "./media"
    matcher = FaceMatcher(media_folder_path)
    img_path = "Sweety.jpg"
    processing_time, similarity, name = matcher.match_face(img_path)
    print(f"Processing Time: {processing_time} seconds")
    print(f"Similarity: {similarity} %")
    print(f"Identity: {name}")
