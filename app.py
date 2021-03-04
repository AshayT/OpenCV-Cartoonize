#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import numpy as np

#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():  

    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            break

        else:
            # converting 3D array to 2D, each row will describe a pixel
            Z = frame.reshape((-1, 3)) 

            # converting to np.float32
            Z = np.float32(Z)

            # defining criteria, number of clusters(K) and applying kmeans()
            interations = 200
            accuracy = 0.9
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, interations, accuracy)
            K = 3 
            ret, label, center = cv2.kmeans(Z, K, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)

            # converting back into uint8
            center = np.uint8(center)

            # making all pixels of centers RBG
            k = [center[i][0] for i in range(3)]
            k2 = np.sort(k)
            for l in range(3):
                i, = np.where(k == k2[l])
                i = int(i)
                center[i][l] = 255
            
            
            #np.random.shuffle(center)
            res = center[label.flatten()]  #giving neighbours colors same as the center
            res2 = res.reshape((frame.shape))

            ret, buffer = cv2.imencode('.jpg', res2) #converting image format into streaming data
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  #concating frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)