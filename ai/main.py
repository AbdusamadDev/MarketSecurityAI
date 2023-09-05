import os
import datetime
import cv2
import numpy as np
import torch
import insightface
import faiss
from imutils.video import VideoStream
from flask import Flask, render_template
from threading import Thread
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
import time

from models import camera_urls


app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app)
CORS(app)


class FaceDetector:
    def __init__(self, root_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"_______________ {self.device}")
        try:
            self.face_model = insightface.app.FaceAnalysis()
            ctx_id = 0
            self.face_model.prepare(ctx_id=ctx_id)
        except Exception as e:
            print(f"Ошибка при инициализации моделей: {e}")
            raise
        
        self.index, self.known_face_names = self.load_face_encodings(root_dir)
        self.emotion = ""
        self.user_id = ""
        self.video_captures = []

    def add_camera(self, urls):
        for url in urls:
            print(f"Adding camera: {url}")
            try:
                video_stream = VideoStream(url).start()
                self.video_captures.append(
                    {"video_stream": video_stream, "camera_url": url}
                )
                print(f"Захват видео для {url} открыт успешно")
            except Exception as e:
                print(f"Ошибка при открытии видеозахвата для {url}: {e}")
                continue

    def start_all_video_captures(self):
        while True:
            for video_capture in self.video_captures:
                for result in self.detect_and_display_faces(**video_capture):
                    yield result

    def load_face_encodings(self, root_dir):
        known_face_encodings = []
        known_face_names = []
        if "media" not in os.listdir(os.getcwd()):
            os.makedirs("media")
        print(f"Processing root directory: {root_dir}")
        print("The dir: ", os.getcwd())
        for dir_name in os.listdir(root_dir):
            print(root_dir)
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                print(f"Processing directory: {dir_path}")
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(".jpg") or file_name.endswith(".png"):
                        image_path = os.path.join(dir_path, file_name)
                        print(f"Processing image: {image_path}")
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
                            box = face.bbox.astype(int)
                            face_image = image[box[1] : box[3], box[0] : box[2]]
                            if face_image.size == 0:
                                continue
                            face_image = cv2.resize(face_image, (640, 480))
                            face_image = face_image / 255.0
                            face_image = (
                                torch.tensor(face_image.transpose((2, 0, 1)))
                                .float()
                                .to(self.device)
                                .unsqueeze(0)
                            )
                            embedding = np.array(face.embedding)
                            known_face_encodings.append(embedding)
                            known_face_names.append(dir_name)
                        except Exception as e:
                            print(f"Unable to process image {image_path}: {e}")
                            continue
        known_face_encodings = np.array(known_face_encodings)
        if known_face_encodings.shape[0] == 0:
            raise ValueError(
                "No face encodings loaded. Please ensure valid images are present."
            )
        self.embeddings = np.array(known_face_encodings)
        self.names = known_face_names
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(self.embeddings)
        return index, self.names

    def recognize_faces(self, frame):
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
            threshold = 400
            if min_distance < threshold:
                identified_name = self.names[best_match_index]
                results.append((identified_name, min_distance, output_path))
                print(
                    f"Identified face as: {identified_name} with distance: {min_distance}"
                )
        return results

    def detect_and_display_faces(self, video_stream, camera_url):
        buffer = []
        last_time = time.time()
        while True:
            try:
                frame = video_stream.read()
                if frame is None:
                    print("Unable to read frame")
                    continue
            except Exception as e:
                print(f"Unable to read frame: {e}")
                continue
            try:
                faces = self.face_model.get(frame)
                if faces is None:
                    print("Model could not process frame")
                    continue
            except Exception as e:
                print(f"Error in face recognition: {e}")
                continue
            if faces:
                recognized_faces = self.recognize_faces(frame)
                if not recognized_faces:
                    print("No recognized faces in this frame.")
                for name, similarity, output in recognized_faces:
                    buffer.append(
                        {
                            "user": name,
                            "similarity": similarity,
                            "image_path": output,
                            "datetime": str(datetime.datetime.now()),
                            "url": camera_url,
                        }
                    )
            current_time = time.time()
            if current_time - last_time >= 1:
                if buffer:
                    names = [result["user"] for result in buffer]
                    most_common_name = max(set(names), key=names.count)
                    most_common_result = next(
                        (
                            result
                            for result in buffer
                            if result["user"] == most_common_name
                        ),
                        None,
                    )
                    yield most_common_result
                    buffer.clear()
                    last_time = current_time


class BackgroundCameraTask(Thread):
    def __init__(self, detector, video_stream):
        self.detector = detector
        self.video_stream = video_stream
        super().__init__()

    def run(self):
        for result in self.detector.detect_and_display_faces(**self.video_stream):
            socketio.emit("response_data", result)


@app.route("/get", methods=["GET"])
@cross_origin()
def get():
    return render_template("index.html")


root_dir = ""
b = str(os.path.abspath(__file__)).split("/")[:-2]
for i in b:
    root_dir += i + "/"
detector = FaceDetector(root_dir=root_dir + "media")
detector.add_camera(camera_urls)
camera_threads = []
for video_capture in detector.video_captures:
    camera_thread = BackgroundCameraTask(detector, video_capture)
    camera_thread.start()
    camera_threads.append(camera_thread)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=11223)
