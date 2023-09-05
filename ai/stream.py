import cv2
import time
import os
from with_faiss import FaceRecognizer

root_dir = "../media"
recognizer = FaceRecognizer(root_dir=root_dir)
cap = cv2.VideoCapture("rtsp://admin:Z12345678r@192.168.0.201/Streaming/channels/2/")
index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_strings = []
    for name, similarity, output in recognizer.recognize_faces(frame):
        output_strings.append(
            f"\nUser: {name}, Similarity: {similarity:.2f}%, Image name: {output}"
        )

    # Print all information for the current frame at once
    print("\n".join(output_strings))

    time.sleep(1)
