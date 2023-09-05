from flask import Flask
from flask_cors import CORS, cross_origin
from flask import Response, stream_with_context
import json
import queue
import os

from background import BackgroundCameraTask


app = Flask(__name__, template_folder="templates")
CORS(app)
results_queue = queue.Queue()


@app.route("/stream", methods=["GET"])
@cross_origin()
def stream():

    def generate():
        while True:
            result = results_queue.get()
            yield json.dumps({"data": str(result) + "\n"})

    return Response(generate(), mimetype="text/plain")


root_dir = os.path.join(*os.path.abspath(__file__).split(os.sep)[:-2], "media")
camera_thread = BackgroundCameraTask(results_queue=results_queue)
camera_thread.start()
app.run(host="0.0.0.0", port=11223)
camera_thread.stop()
