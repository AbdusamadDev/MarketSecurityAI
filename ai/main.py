from flask_cors import CORS, cross_origin
from flask import Response
from flask import Flask
import queue
import json

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
            print(result)
            yield json.dumps({"data": str(result) + "\n"})

    return Response(generate(), mimetype="text/plain")


camera_thread = BackgroundCameraTask(results_queue=results_queue)
# camera_thread.start()
app.run(host="0.0.0.0", port=11223)
