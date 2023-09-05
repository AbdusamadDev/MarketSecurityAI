import os
import datetime
import cv2
import numpy as np
import torch
import insightface
import faiss
from imutils.video import VideoStream
from flask import Flask, render_template
from threading import Thread, Event
from collections import Counter
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
        self.face_model = self._initialize_face_model()
        (
            self.index,
            self.segment_face_encodings,
            self.known_face_names,
        ) = self.load_face_encodings(root_dir)
        self.video_captures = []
        self.face_last_seen = None

    def _initialize_face_model(self):
        try:
            model = insightface.app.FaceAnalysis()
            ctx_id = 0
            model.prepare(ctx_id=ctx_id)
            return model
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def add_camera(self, urls):
        for url in urls:
            try:
                video_stream = VideoStream(url).start()
                self.video_captures.append(
                    {"video_stream": video_stream, "camera_url": url}
                )
            except Exception as e:
                print(f"Error opening video capture for {url}: {e}")

    def start_all_video_captures(self):
        while True:
            for video_capture in self.video_captures:
                for result in self.recognition(**video_capture):
                    yield result

    def load_face_encodings(self, root_dir):
        known_face_encodings = []
        segment_face_encodings = {"top": [], "bottom": [], "left": [], "right": []}
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

                            # Compute segment embeddings
                            h, w, _ = face_image.shape
                            segments = {
                                "top": face_image[: h // 2, :],
                                "bottom": face_image[h // 2 :, :],
                                "left": face_image[:, : w // 2],
                                "right": face_image[:, w // 2 :],
                            }

                            for segment, segment_img in segments.items():
                                # Process segment_img similar to face_image
                                segment_embedding = np.array(
                                    face.embedding
                                )  # Compute embedding for segment_img
                                segment_face_encodings[segment].append(
                                    segment_embedding
                                )

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
        return index, segment_face_encodings, known_face_names

    def recognize_faces(self, frame, segment):
        results = []
        if not os.path.exists("cutted"):
            os.makedirs("cutted")
        faces = self.face_model.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            output_path = os.path.join("cutted", f"{time.time()}.jpg")
            cv2.imwrite(output_path, frame[bbox[1] : bbox[3], bbox[0] : bbox[2]])
            embedding_new = np.array(face.embedding)
            distances = np.linalg.norm(
                self.segment_face_encodings[segment] - embedding_new, axis=1
            )
            best_match_index = np.argmin(distances)
            min_distance = distances[best_match_index]
            threshold = 300
            if min_distance < threshold:
                identified_name = self.names[best_match_index]
                results.append((identified_name, min_distance, output_path))
                print(
                    f"Identified face as: {identified_name} with distance: {min_distance}"
                )
        return results

    def recognition(self):
        buffer = []
        last_time = time.time()
        face_present = False
        for video_capture in self.video_captures:
            frame = video_capture["video_stream"].read()
            if frame is None:
                continue

            self.frame_shape = frame.shape  # Store the frame shape

            faces = self.face_model.get(frame)
            current_time = time.time()
            if faces:
                for face in faces:
                    segment = self._determine_segment(face.bbox.astype(int))
                    recognized_faces = self.recognize_faces(frame, segment)
                face_present = True
                buffer.extend(recognized_faces)
            else:
                face_present = False

            if current_time - last_time >= 1:
                if buffer:
                    if (
                        self.face_last_seen is None
                        or current_time - self.face_last_seen > 1
                    ):
                        most_common_name, _ = Counter(
                            [result["user"] for result in buffer]
                        ).most_common(1)[0]
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

            if face_present:
                self.face_last_seen = current_time
            else:
                self.face_last_seen = None

    def _determine_segment(self, bbox):
        (
            frame_height,
            frame_width,
            _,
        ) = self.frame_shape
        x, y, w, h = bbox

        if y + h < frame_height // 2:
            return "top"
        elif y > frame_height // 2:
            return "bottom"
        elif x + w < frame_width // 2:
            return "left"
        else:
            return "right"


class BackgroundCameraTask(Thread):
    def __init__(self, detector):
        self.detector = detector
        self.stop_event = Event()
        super().__init__()

    def run(self):
        while not self.stop_event.is_set():
            for result in self.detector.recognition():
                socketio.emit("response_data", result)

    def stop(self):
        self.stop_event.set()


@app.route("/get", methods=["GET"])
@cross_origin()
def get():
    return render_template("index.html")


if __name__ == "__main__":
    # root_dir = os.path.join(*os.path.abspath(__file__).split(os.sep)[:-2], "media")
    root_dir = "/home/ocean/Projects/MarketPlaceSecurityApp/media"
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Root directory: ", root_dir)
    detector = FaceDetector(root_dir=root_dir)
    detector.add_camera(camera_urls)
    camera_thread = BackgroundCameraTask(detector)
    camera_thread.start()
    socketio.run(app, host="0.0.0.0", port=11223)
    camera_thread.stop()
