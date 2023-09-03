import os
import cv2
import numpy as np
import insightface
import torch
from facenet_pytorch import InceptionResnetV1


class FaceRecognizerFaceNet:
    def __init__(self, root_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_model = insightface.app.FaceAnalysis()
        ctx_id = 0
        self.face_model.prepare(ctx_id=ctx_id)

        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.known_face_encodings, self.known_face_names = self.load_face_encodings(
            root_dir
        )

    def load_face_encodings(self, root_dir):
        known_face_encodings = []
        known_face_names = []
        for dir_name in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.endswith((".jpg", ".png")):
                        image_path = os.path.join(dir_path, file_name)
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        faces = self.face_model.get(image)
                        if not faces:
                            continue
                        face = faces[0]
                        face_bbox = face.bbox.astype(int)
                        face_img = image[
                            face_bbox[1] : face_bbox[3], face_bbox[0] : face_bbox[2]
                        ]
                        img_cropped = cv2.resize(face_img, (160, 160))
                        img_cropped = (
                            torch.Tensor(img_cropped)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        embedding = (
                            self.facenet(img_cropped).detach().cpu().numpy().squeeze()
                        )
                        known_face_encodings.append(embedding)
                        known_face_names.append(dir_name)
        return known_face_encodings, known_face_names

    def recognize_faces(self, frame):
        faces = self.face_model.get(frame)
        results = []
        if faces:
            for face in faces:
                face_bbox = face.bbox.astype(int)
                face_img = frame[
                    face_bbox[1] : face_bbox[3], face_bbox[0] : face_bbox[2]
                ]
                img_cropped = cv2.resize(face_img, (160, 160))
                img_cropped = (
                    torch.Tensor(img_cropped)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self.device)
                )
                embedding = self.facenet(img_cropped).detach().cpu().numpy().squeeze()

                distances = np.linalg.norm(
                    self.known_face_encodings - embedding, axis=1
                )
                idx_min = distances.argmin()
                similarity_percentage = max(0, 1 - distances[idx_min]) * 100
                name = self.known_face_names[idx_min]
                results.append((name, similarity_percentage))
        return results


if __name__ == "__main__":
    root_dir = "../media"
    recognizer = FaceRecognizerFaceNet(root_dir=root_dir)

    image_path = "Ilyosxon.png"
    frame = cv2.imread(image_path)
    for name, similarity, processing_time in recognizer.recognize_faces(frame):
        print(
            f"\n\nDetected face: {name}, Similarity: {similarity:.2f}%, Processing Time: {processing_time:.4f} seconds\n"
        )
