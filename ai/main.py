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

        try:
            self.index, self.known_face_names = self.load_face_encodings(root_dir)
        except Exception as e:
            print(f"Ошибка при загрузке кодировок лиц: {e}")
            raise

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
        print("Known face encodings shape:", known_face_encodings.shape)

        if known_face_encodings.shape[0] == 0:
            raise ValueError(
                "No face encodings loaded. Please ensure valid images are present."
            )

        index = faiss.IndexFlatL2(known_face_encodings.shape[1])
        index.add(known_face_encodings)

        return index, known_face_names

    def detect_and_display_faces(self, video_stream, camera_url):
        names = []
        screenshot_interval = 0.3
        last_screenshot_time = datetime.datetime.now()

        while True:
            current_time = datetime.datetime.now()

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
                for face in faces:
                    box = face.bbox.astype(int)
                    face_image = frame[box[1] : box[3], box[0] : box[2]]
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
                    embedding = face.embedding

                    D, I = self.index.search(embedding.reshape(1, -1), 1)
                    if D[0, 0] < 600:
                        name = self.known_face_names[I[0, 0]]
                        names.append(name)
                        yield {
                            "user": name,
                            "datetime": str(datetime.datetime.now()),
                            "url": camera_url,
                        }
            if (
                current_time - last_screenshot_time
            ).total_seconds() > screenshot_interval:
                screenshot_filename = os.path.join(
                    "./media",
                    f"{current_time.strftime('%Y_%m_%d_%H_%M_%S')}.jpg",
                )
                cv2.imwrite(screenshot_filename, frame)
                last_screenshot_time = current_time


class BackgroundCameraTask(Thread):
    def __init__(self, detector, video_stream):
        self.detector = detector
        self.video_stream = video_stream
        super().__init__()

    def run(self):
        for result in self.detector.detect_and_display_faces(**self.video_stream):
            print(result)
            socketio.emit("response_data", result)


@app.route("/get", methods=["GET"])
@cross_origin()
def get():
    return render_template("index.html")


@app.route("/hello")    
@cross_origin()
def hello():
    return "asdkjlhasdfljkhkjlasdf"


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


# rtsp://admin:Z12345678r@192.168.0.201/Streaming/channels/2/
# 40.99695646
# 71.64007184
