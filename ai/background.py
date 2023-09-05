from threading import Thread, Event

from models import camera_urls
from recognition import FaceDetector


class BackgroundCameraTask(Thread):
    def __init__(self, results_queue):
        self.results_queue = results_queue
        self.root_dir = "/home/ocean/Projects/MarketPlaceSecurityApp/media"
        self.detector = FaceDetector(root_dir=self.root_dir)
        self.detector.add_camera(camera_urls)
        self.stop_event = Event()
        super().__init__()

    def run(self):
        while not self.stop_event.is_set():
            for result in self.detector.recognition():
                print("Result in background: ", result)
                self.results_queue.put(result)

    def stop(self):
        self.stop_event.set()
