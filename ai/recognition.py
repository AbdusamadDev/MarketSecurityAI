import cv2
import numpy as np
import torch
import insightface
import faiss
import os
from imutils.video import VideoStream
import asyncio

import time


class FaceDetector:
    def __init__(self, root_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_model = self._initialize_face_model()
        self.index, self.known_face_names = self.load_face_encodings(root_dir)
        self.video_captures = []

    def _initialize_face_model(self):
        model = insightface.app.FaceAnalysis()
        ctx_id = 0
        model.prepare(ctx_id=ctx_id)
        return model

    async def add_camera(self, urls):
        tasks = [self._add_single_camera(url) for url in urls]
        await asyncio.gather(*tasks)

    async def _add_single_camera(self, url):
        video_stream = VideoStream(url).start()
        self.video_captures.append({"video_stream": video_stream, "camera_url": url})

    def load_face_encodings(self, root_dir):
        user_embeddings = {}  # Dictionary to store embeddings for each user
        known_face_names = []

        if "media" not in os.listdir(os.getcwd()):
            os.makedirs("media")

        for dir_name in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                embeddings = []  # Store embeddings for all images of a user
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(".jpg") or file_name.endswith(".png"):
                        image_path = os.path.join(dir_path, file_name)
                        try:
                            image = cv2.imread(image_path)
                            if image is None:
                                print(f"Unable to read image: {image_path}")
                                continue
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            faces = self.face_model.get(image)
                            if not faces:
                                print(f"No faces detected in image: {image_path}")
                                continue
                            face = faces[0]
                            embedding = np.array(face.embedding)
                            embeddings.append(embedding)
                        except Exception as e:
                            print(f"Unable to process image {image_path}: {e}")
                            continue
                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)  # Average embedding
                    user_embeddings[dir_name] = avg_embedding
                    known_face_names.append(dir_name)

        # Convert embeddings to a suitable format for faiss
        known_face_encodings = np.array(list(user_embeddings.values()))

        if known_face_encodings.shape[0] == 0:
            raise ValueError(
                "No face encodings loaded. Please ensure valid images are present."
            )
        self.embeddings = known_face_encodings
        self.names = known_face_names

        index = faiss.IndexFlatL2(known_face_encodings.shape[1])
        index.add(known_face_encodings)
        return index, known_face_names

    def _sync_recognize_faces(self, frame, camera_url):
        results = []
        if not os.path.exists("cutted"):
            os.makedirs("cutted")
        faces = self.face_model.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            output_path = os.path.join("cutted", f"{time.time()}.jpg")
            cv2.imwrite(output_path, frame[bbox[1] : bbox[3], bbox[0] : bbox[2]])
            embedding_new = np.array(face.embedding)
            distances = np.linalg.norm(self.embeddings - embedding_new, axis=1)
            best_match_index = np.argmin(distances)
            min_distance = distances[best_match_index]
            threshold = 200
            if min_distance < threshold:
                identified_name = self.names[best_match_index]
                results.append(
                    {
                        "user": identified_name,
                        "distance": min_distance,
                        "image_path": output_path,
                        "camera_url": camera_url,
                    }
                )
                print(
                    f"Identified face as: {identified_name} with distance: {min_distance}"
                )
        return results

    async def _async_recognize_faces(self, frame, camera_url):
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._sync_recognize_faces, frame, camera_url
        )
        return results

    async def _process_single_camera(self, video_capture):
        frame = video_capture["video_stream"].read()
        if frame is not None:
            recognized_faces = await self._async_recognize_faces(
                frame, video_capture["camera_url"]
            )
            return recognized_faces
        return []

    async def async_recognition(self):
        while True:
            tasks = [
                self._process_single_camera(video_capture)
                for video_capture in self.video_captures
            ]
            results = await asyncio.gather(*tasks)

            all_faces = []
            for face_list in results:
                all_faces.extend(face_list)
            for face in all_faces:
                yield face
