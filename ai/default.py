import os
import cv2
import numpy as np
import insightface
import torch
import faiss
import time


class FaceRecognizer:
    def __init__(self, root_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_model = insightface.app.FaceAnalysis()
        ctx_id = 0
        self.face_model.prepare(ctx_id=ctx_id)

        self.index, self.known_face_names = self.load_face_encodings(root_dir)

    def load_face_encodings(self, root_dir):
        known_face_encodings = []
        known_face_names = []

        for dir_name in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(".jpg") or file_name.endswith(".png"):
                        image_path = os.path.join(dir_path, file_name)
                        try:
                            image = cv2.imread(image_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            faces = self.face_model.get(image)
                            if not faces:
                                continue

                            face = faces[0]
                            embedding = np.array(face.embedding)
                            known_face_encodings.append(embedding)
                            known_face_names.append(dir_name)
                        except Exception as e:
                            continue

        known_face_encodings = np.array(known_face_encodings)
        index = faiss.IndexFlatL2(known_face_encodings.shape[1])
        index.add(known_face_encodings)

        return index, known_face_names

    def recognize_faces(self, frame):
        start_time = time.time()
        faces = self.face_model.get(frame)
        end_time = time.time()
        processing_time = end_time - start_time

        if faces:
            for face in faces:
                embedding = face.embedding
                D, I = self.index.search(embedding.reshape(1, -1), 1)
                similarity_percentage = (1 - D[0, 0] / 600) * 100
                name = self.known_face_names[I[0, 0]]
                yield name, similarity_percentage, processing_time


if __name__ == "__main__":
    root_dir = "../media"
    recognizer = FaceRecognizer(root_dir=root_dir)

    image_path = "Ilyosxon.png"
    frame = cv2.imread(image_path)
    for name, similarity, processing_time in recognizer.recognize_faces(frame):
        print(
            f"\n\nDetected face: {name}, Similarity: {similarity:.2f}%, Processing Time: {processing_time:.4f} seconds\n"
        )
