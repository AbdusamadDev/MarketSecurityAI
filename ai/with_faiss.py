import os
import cv2
import numpy as np
import insightface
import faiss
import time


class FaceRecognizer:
    def __init__(self, root_dir):
        self.face_model = insightface.app.FaceAnalysis()
        ctx_id = 0
        self.face_model.prepare(ctx_id=ctx_id)
        self.index, self.known_face_names = self.load_face_encodings(root_dir)

    def load_face_encodings(self, root_dir):
        known_face_encodings = []
        known_face_names = []

        try:
            known_face_encodings = np.load("embeddings.npy")
            with open("names.txt", "r") as f:
                known_face_names = [line.strip() for line in f]
        except (FileNotFoundError, IOError):
            for dir_name in os.listdir(root_dir):
                dir_path = os.path.join(root_dir, dir_name)
                if os.path.isdir(dir_path):
                    for file_name in os.listdir(dir_path):
                        if file_name.endswith(".jpg") or file_name.endswith(".png"):
                            image_path = os.path.join(dir_path, file_name)
                            embedding, name = self.process_image(image_path, dir_name)
                            if embedding is not None:
                                known_face_encodings.append(embedding)
                                known_face_names.append(name)
            # Convert and save
            known_face_encodings = np.array(known_face_encodings)
            np.save("embeddings.npy", known_face_encodings)
            with open("names.txt", "w") as f:
                for name in known_face_names:
                    f.write(name + "\n")

        # Faiss setup
        faiss.normalize_L2(known_face_encodings)
        index = faiss.IndexFlatIP(known_face_encodings.shape[1])
        index.add(known_face_encodings)

        return index, known_face_names

    def process_image(self, image_path, dir_name):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_model.get(image)
        if not faces:
            return None, None
        face = faces[0]
        embedding = np.array(face.embedding)
        return embedding, dir_name

    def recognize_faces(self, frame):
        if not os.path.exists("cutted"):
            os.makedirs("cutted")

        faces = self.face_model.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            output_path = os.path.join("cutted", f"{time.time()}.jpg")
            cv2.imwrite(output_path, frame[bbox[1] : bbox[3], bbox[0] : bbox[2]])
            embedding = np.array(face.embedding)
            faiss.normalize_L2(embedding.reshape(1, -1))
            D, I = self.index.search(embedding.reshape(1, -1), 1)
            cosine_similarity = D[0, 0]
            similarity_percentage = ((cosine_similarity + 1) / 2) * 100
            name = self.known_face_names[I[0, 0]]
            return name, similarity_percentage, output_path
