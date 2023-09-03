import cv2


class BaseApp:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(self.url)

    def is_opened(self):
        return self.cap.isOpened()

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()


class VideoPlayer:
    def __init__(self, video_streamer):
        self.video_streamer = video_streamer

    def play(self):
        while True:
            ret, frame = self.video_streamer.read_frame()
            if not ret:
                print("Can't receive frame. Exiting ...")
                break
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    def close(self):
        cv2.destroyAllWindows()
        self.video_streamer.release()


def main():
    cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if cuda_enabled:
        print("OpenCV was built with CUDA support!")
    else:
        print("OpenCV was not built with CUDA support.")

    video_streamer = BaseApp("")
    if not video_streamer.is_opened():
        print("Failed to open video stream.")
        return

    video_player = VideoPlayer(video_streamer)
    video_player.play()
    video_player.close()


if __name__ == "__main__":
    main()