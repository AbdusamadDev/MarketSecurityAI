import os
import cv2
import numpy as np
import insightface
import faiss
import time


class FaceRecognizer:
    def __init__(self, root_dir):
        self.face_model = insightface.app.FaceAnalysis()
        ctx_id = 0  # Set to the first GPU
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
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        faces = self.face_model.get(image)
                        if not faces:
                            continue
                        face = faces[0]
                        embedding = np.array(face.embedding)
                        known_face_encodings.append(embedding)
                        known_face_names.append(dir_name)

        known_face_encodings = np.array(known_face_encodings)
        faiss.normalize_L2(known_face_encodings)

        # Prepare for GPU usage in FAISS
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(known_face_encodings.shape[1])
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(known_face_encodings)

        return gpu_index, known_face_names

    def recognize_faces(self, frame):
        faces = self.face_model.get(frame)
        for face in faces:
            embedding = np.array(face.embedding)
            faiss.normalize_L2(embedding.reshape(1, -1))
            D, I = self.index.search(embedding.reshape(1, -1), 1)

            cosine_similarity = D[0, 0]
            similarity_percentage = ((cosine_similarity + 1) / 2) * 100
            name = self.known_face_names[I[0, 0]]

            bbox = face.bbox.astype(int)
            cropped_face = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]

            output_path = os.path.join("cutted", f"{time.time()}.jpg")
            cv2.imwrite(output_path, cropped_face)

            yield name, similarity_percentage, output_path


if __name__ == "__main__":
    root_dir = "../media"
    recognizer = FaceRecognizer(root_dir=root_dir)
    image_path = "media/i.jpg"
    frame = cv2.imread(image_path)

    start_time = time.time()
    results = list(recognizer.recognize_faces(frame))
    end_time = time.time()

    for name, similarity, output in results:
        print(
            f"Detected face: {name}, Similarity: {similarity:.2f}%\nImage name: {output}"
        )
    print(f"Face recognition took: {end_time - start_time:.4f} seconds")
