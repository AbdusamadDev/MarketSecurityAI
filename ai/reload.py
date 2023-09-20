import os
import time
import cv2
import numpy as np
import torch
import insightface
import faiss


class FaceDetector:
    def __init__(self, root_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.face_model = insightface.app.FaceAnalysis()
            ctx_id = 0
            self.face_model.prepare(ctx_id=ctx_id)
        except Exception as e:
            print(f"Error during model initialization: {e}")
            raise

        try:
            self.index, self.known_face_names = self.load_face_encodings(root_dir)
        except Exception as e:
            print(f"Error loading face encodings: {e}")
            raise

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
                        except Exception as e:
                            print(f"Unable to process image {image_path}: {e}")
                            continue

                        if faces:
                            face = faces[0]
                            embedding = face.embedding
                            known_face_encodings.append(embedding)
                            known_face_names.append(dir_name)

        known_face_encodings = np.array(known_face_encodings)
        index = faiss.IndexFlatL2(known_face_encodings.shape[1])
        index.add(known_face_encodings)
        return index, known_face_names

    def process_image(self, image_path):
        start_time = time.time()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_model.get(image)

        if not faces:
            print("No faces detected.")
            return

        face = faces[0]
        embedding = face.embedding
        D, I = self.index.search(embedding.reshape(1, -1), 1)

        elapsed_time = time.time() - start_time
        name = self.known_face_names[I[0, 0]]
        similarity = D[0, 0]

        print(f"Detected person: {name}")
        print(f"Similarity percentage: {similarity}%")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")


# Example of usage:
root_dir = "./media"
face_detector = FaceDetector(root_dir)
face_detector.process_image("Sweety.jpg")
