# Deploy app : scp -r ./* wattana@192.200.9.215:/home/wattana/flask_bruiseduck

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit, send
from flask.json import jsonify
import time
import io
import base64
import cv2
import numpy as np
import tensorflow as tf
import imutils
import pandas as pd

import json
from bson import ObjectId

from PIL import Image
from flask_cors import CORS, cross_origin
from engineio.payload import Payload
from object_detection.utils import label_map_util
from workspace.tf_inference import show_inferencev3
from imutils.video import FileVideoStream
from imutils.video import FPS

Payload.max_decode_packets = 2048

# Load models
detection_model = tf.saved_model.load("./workspace/models/saved_model/")
PATH_TO_LABELS = './workspace/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

app = Flask(__name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static")
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins='*')

global fps, prev_frame_time, new_frame_time
fps = 0
prev_frame_time = 0
new_frame_time = 0

def gen_frames():
    # capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture("./videos/Back_25.mp4")
    capture = cv2.VideoCapture("../../datasets/bruiseduck/videos/Back_04.mp4")

    global fps, prev_frame_time, new_frame_time

    while True:
        # capture.set(cv2.CAP_PROP_FPS, 25)
        success, frame = capture.read()  # read the camera frame
        if not success:
            break
        else:
            # print(capture.get(cv2.CAP_PROP_FPS))
            frame = cv2.resize(frame, (640, 480))
            (h, w, c) = frame.shape[:3]

            xW1 = int(w * (1.25/4))    
            xW2 = int(w * (2.75/4)) 
            xW3 = int(w * (1.7/4))

            # Font which we will be using to display FPS
            wW = 7; hH = 25
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            text = 'FPS: '+str(int(fps))

            # Rectangle
            frame = cv2.rectangle(frame, (0, 0), (xW1, h), (0, 0, 0), thickness=-1)
            frame = cv2.rectangle(frame, (xW2, 0), (w, h), (0, 0, 0), thickness=-1)
            frame = cv2.rectangle(frame, (xW1, 0), (xW2, h), (0, 255, 0), thickness=2)

            # Predict
            frame, code = show_inferencev3(detection_model, category_index, frame, min_score_thresh=0.5)

            # Display FPS on frame
            cv2.putText(frame, text, (wW, hH * 2), font, 0.75, (50, 170, 50), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    capture.release()
    cv2.destroyAllWindows()

def gen_frames_v2():
    # capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture("./videos/Back_25.mp4")
    capture = cv2.VideoCapture("../../datasets/bruiseduck/videos/Back_04.mp4")

    # Exit if video not opened.
    if not capture.isOpened():
        print("Could not open video")
        # sys.exit()

    # Read first frame.
    ok, frame = capture.read()
    if not ok:
        print ('Cannot read video file')
        # sys.exit()


    global fps, prev_frame_time, new_frame_time
    fps = 0
    prev_frame_time = 0
    new_frame_time = 0    

    while True:
        # Font which we will be using to display FPS
        wW = 7; hH = 25
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        text = 'FPS: '+str(int(fps))

        ok, frame = capture.read()

        # print(capture.get(cv2.CAP_PROP_FPS))
        frame = cv2.resize(frame, (640, 480))
        (h, w, c) = frame.shape[:3]

        xW1 = int(w * (1.25/4))    
        xW2 = int(w * (2.75/4)) 
        xW3 = int(w * (1.7/4))
        
        # Convert color
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        # Rectangle
        frame = cv2.rectangle(frame, (0, 0), (xW1, h), (0, 0, 0), thickness=-1)
        frame = cv2.rectangle(frame, (xW2, 0), (w, h), (0, 0, 0), thickness=-1)
        frame = cv2.rectangle(frame, (xW1, 0), (xW2, h), (0, 255, 0), thickness=2)

        # Predict
        frame, code = show_inferencev3(detection_model, category_index, frame, min_score_thresh=0.5)

        # Display FPS on frame
        cv2.putText(frame, text, (wW, hH * 2), font, 0.75, (50, 170, 50), 2)

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    capture.release()
    cv2.destroyAllWindows()

def gen_frames_v3():
    # capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture("./static/videos/Back_25.mp4")
    capture = cv2.VideoCapture("../../datasets/bruiseduck/videos/Back_04.mp4")

    # Exit if video not opened.
    if not capture.isOpened():
        print("Could not open video")
        # sys.exit()

    # Read first frame.
    ok, frame = capture.read()
    if not ok:
        print ('Cannot read video file')
        # sys.exit()


    global count, fps, prev_frame_time, new_frame_time
    count = 0; fps = 0; prev_frame_time = 0; new_frame_time = 0;    

    df_detect = pd.DataFrame({'counter': [], 'code':[]})
    check = True

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    
    size = (frame_width, frame_height)
    print(size )

    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter('bruiseduck.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 24, size)

    while True:
        # Font which we will be using to display FPS
        wW = 7; hH = 25
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        text = 'FPS: '+str(int(fps))

        ok, frame = capture.read()

        # print(capture.get(cv2.CAP_PROP_FPS))
        # frame = cv2.resize(frame, (640, 480))
        (h, w, c) = frame.shape[:3]

        xW1 = int(w * (1.2/4))    
        xW2 = int(w * (2.8/4)) 
        xW3 = int(w * (2/4))
        
        # Convert color to RGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        # Rectangle
        frame = cv2.rectangle(frame, (0, 0), (xW1, h), (0, 0, 0), thickness=-1)
        frame = cv2.rectangle(frame, (xW2, 0), (w, h), (0, 0, 0), thickness=-1)
        # frame = cv2.rectangle(frame, (xW1, 0), (xW2, h), (0, 255, 0), thickness=2)

        # Predict
        frame, code, score, box = show_inferencev3(detection_model, category_index, frame, min_score_thresh=0.5)

        # Line
        cv2.line(frame, (xW3, 0),(xW3, h),(255, 0 ,0))   

        # Get bounding box from prediction output
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
        # print(left, right, top, bottom)
    
        p1 = (int(left), int(right))
        p2 = (int(top), int(bottom))

        # Draw the bounding box
        bX,bY = p1
        bW,bH = p2

        # Draw circle in the center of the bounding box
        bX1 = bX + int(bX/2)
        bY1 = bY + int(bY/2)
        
        bX2 = int((bX + bY) / 2)
        bY2 = int((bW + bH) / 2)

        if sum(box) > 0:
            # frame = cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255))
            cv2.circle(frame, (bX2, bY2), 1, (255, 0 ,0), 2)

            if (bX2 > xW3):
                if(check):
                    count += 1
                    check = False
            else:
                check = True     

        # Display FPS on frame
        cv2.putText(frame, text, (wW, hH * 2), font, 0.75, (50, 170, 50), 2)

        # Display Counter
        cv2.putText(frame, "Counter: " + str(count), (wW, hH * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Print the centroid coordinates (we'll use the center of the bounding box) on the image
        center = "x: " + str(bX2) + ", y: " + str(bY2)
        cv2.putText(frame, str(center), (wW, hH * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        
        if (code == "FP09" or code == "FP10" or code == "FP11" or code == "FP16" or code == "FP22" or code == "FP32"):
          
            if(len(df_detect) > 0):
                listText = [[str(count),str(code)]]
                listText = [tuple(i) for i in listText]
                checkDuplicate = df_detect[df_detect[['counter', 'code']].apply(tuple, axis = 1).isin(listText)]

                if(len(checkDuplicate) == 0):
                    print("counter: ", count, "code: ", code, "score: ", score)
                    df_detect = df_detect.append({'counter': str(count), 'code': str(code)}, ignore_index = True).drop_duplicates().reset_index(drop=True)

            else:
                print("counter: ", count, "code: ", code, "score: ", score)  
                df_detect = df_detect.append({'counter': str(count), 'code': str(code)}, ignore_index = True).drop_duplicates().reset_index(drop=True)

        # Convert color to BGR
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)    

        # Write the frame into the
        # file 'filename.avi'
        # result.write(frame)    

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def moving_average(x):
    return np.mean(x)

@app.route('/flask_bruiseduck/', methods=['POST', 'GET'])
def index():
    return render_template('base.html', mimetype='text/html')

@app.route('/flask_bruiseduck/detect', methods=['POST', 'GET'])
def detect():
    return render_template('./pages/detect.html')

@app.route('/flask_bruiseduck/detect_video', methods=['POST', 'GET'])
def detect_video():
    return render_template('./pages/detect_video.html')    

@app.route('/flask_bruiseduck/report', methods=['POST', 'GET'])
def report():
    return render_template('./pages/report.html')

@app.route('/flask_bruiseduck/', methods=['POST', 'GET'])
def index_server():
    # return render_template('index_server.html')
    return render_template('base.html')

@app.route('/flask_bruiseduck/video', methods=['GET'])
def index_local():
      return render_template('index_server.html')
#     return render_template('index_local.html')

@app.route('/flask_bruiseduck/video_feed/', methods=["GET"])
def video_feed():
    return Response(gen_frames_v3(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('image')
def image(data_image, truckno):
    global fps, prev_frame_time, new_frame_time
    frame = (readb64(data_image))

    # font which we will be using to display FPS
    wW = 7
    hH = 25
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    text = 'FPS: '+str(int(fps))

    # Convert color
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Predict
    frame, code, score, box = show_inferencev3(detection_model, category_index, frame, min_score_thresh=0.5, truckno=truckno)

    # Display FPS on frame
    cv2.putText(frame, text, (wW, hH * 2), font, 0.75, (50,170,50), 2);

    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

@socketio.on("message")
def handleMessage(msg):
    print(msg)
    send(msg, broadcast=True)
    return None

@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
