from threading import Thread, Event
import asyncio
import os

from recognition import FaceDetector
from models import camera_urls


class BackgroundCameraTask(Thread):
    def __init__(self, results_queue):
        super().__init__()  # Call the parent class constructor first
        self.results_queue = results_queue
        self.root_dir = os.path.join(
            *os.path.abspath(__file__).split(os.sep)[:-2], "media"
        )
        self.detector = FaceDetector(root_dir="/" + str(self.root_dir))
        asyncio.run(self.detector.add_camera(camera_urls))
        self.stop_event = Event()
        self.start()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.async_run())
        loop.close()

    async def async_run(self):
        async for result in self.detector.async_recognition():
            self.results_queue.put(result)
