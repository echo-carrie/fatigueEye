# imports
import base64
import json

import torch
from flask import Response, Flask, render_template, url_for, session
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from datetime import date, datetime
import numpy as np
import threading
import imutils
import random
import playsound
import math
import time
import dlib
import cv2
import io

from EAR import eye_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from MAR import mouth_aspect_ratio
# from EAR import eye_aspect_ratio
# from HeadPose import getHeadTiltAndCoords
# from MAR import mouth_aspect_ratio
from imutils import face_utils
import argparse
import imutils
import time
import math
import cv2
import torch
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'Mi6gttkkSJHof5-q8-HPBUyTsdRVVOLO'

outputFrame = None
lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/video')
def video():
    return render_template('video.html', title='Video')

def sound_alarm():
    playsound.playsound('./fatigue/sounds/alarm2.mp3')
#
def generate():
    # 初始化 dlib 的面部检测器（基于 HOG），然后创建
    # 面部地标预测器
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

    # 初始化视频流并休眠一会儿，让摄像机传感器预热
    print("[INFO] initializing camera...")

    video_file_path = r"F:\数据集\YawDD.rar\YawDD\YawDD dataset\Dash\Female\4-FemaleNoGlasses.avi"
    # vs = cv2.VideoCapture(video_file_path)
    vs = cv2.VideoCapture(0)
    # vs =  VideoStream(src="F:\数据集\YawDD.rar\YawDD\YawDD dataset\Dash\Male\2-MaleGlasses.avi").start()
    # vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
    time.sleep(2.0)
    model = torch.hub.load('./', 'custom', path='./face.pt', source='local')
    # 设置阈值
    model.conf = 0.52  # confidence threshold (0-1)
    model.iou = 0.45  # NMS IoU threshold (0-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # 400x225 to 1024x576
    frame_width = 1024
    frame_height = 576
    Roll = 0
    Rolleye = 0
    Rollmouth = 0
    detectedValues = {
        'head_tilt_degree': 0,
        'perclos': 0,
        'mar': 0,
        'ear': 0,
    }
    # 循环播放视频流中的帧
    # 2D 图像点。如果更改图像，则需要更改矢量
    image_points = np.array([
        (359, 391),  # Nose tip 34
        (399, 561),  # Chin 9
        (337, 297),  # Left eye left corner 37
        (513, 301),  # Right eye right corne 46
        (345, 465),  # Left Mouth corner 49
        (453, 469)  # Right mouth corner 55
    ], dtype="double")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # 设置阈值
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.79
    EYE_AR_CONSEC_FRAMES = 3
    COUNTER = 0

    # 为嘴巴获取面部地标索引
    (mStart, mEnd) = (49, 68)

    while True:
        # 从线程视频流中抓取帧，调整其大小至
        # 最大宽度为 400 像素，并将其转换为
        # 灰度

        ret, frame = vs.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=640, height=576)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(gray)
        results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
        pd1 = results.xyxy[0]
        pd = results.pandas().xyxy[0]

        person_list = pd[pd['name'] == 'face'].to_numpy()
        size = gray.shape

        if not pd.empty and 'xmin' in pd.columns and not pd['xmin'].isnull().all():
            xmin = int(pd['xmin'].values[0])
            ymin = int(pd['ymin'].values[0])
            xmax = int(pd['xmax'].values[0])
            ymax = int(pd['ymax'].values[0])
            # 其他操作...
        else:
            # 处理空列的情况或者跳过当前循环
            pass
            # detect faces in the grayscale frame
        rects = [dlib.rectangle(left=xmin, top=ymin, right=xmax, bottom=ymax)]

        # 检查是否检测到人脸，如果是，则绘制帧上人脸的总数。
        # 帧上的人脸数量
        if rects is not None:
            text = "{} face(s) found".format(len(rects))
            cv2.putText(frame, text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 循环检测人脸
        for rect in rects:
            # 计算面的包围盒并将其绘制在
            # 边框
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 确定面部区域的面部地标，然后
            # 将面部地标 (x, y) 坐标转换为 NumPy数组
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # 将两只眼睛的长宽比平均到一起
            ear = (leftEAR + rightEAR) / 2.0

            # 计算左眼和右眼的凸壳，然后
            # 可视化每只眼睛
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # 计算左眼和右眼的凸壳，然后
            # 可视化每只眼睛
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                Rolleye += 1
                # 如果闭眼的次数足够多
                # 则显示警告
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "Eyes Closed!", (500, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 否则，眼球长宽比不会低于眨眼
                # 阈值，因此重置计数器和警报
            else:
                COUNTER = 0

            mouth = shape[mStart:mEnd]

            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            # 计算嘴巴的凸壳，然后
            # 可视化嘴巴
            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 如果嘴巴张开，则绘制文字
            if mar > MOUTH_AR_THRESH:
                Rollmouth += 1
                cv2.putText(frame, "Yawning!", (800, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            Roll += 1
            # 当检测满150帧时，计算模型得分
            perclos = (Rolleye / Roll) + (Rollmouth / Roll) * 0.2
            # 在前端UI输出perclos值
            cv2.putText(frame, "Perclos: {:.2f}".format(perclos), (650, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if perclos > 0.38:
                cv2.putText(frame, "tired".format(perclos), (800, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "alert".format(perclos), (800, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if Roll == 150:
                # 归零
                # 将三个计数器归零
                # 重新开始新一轮的检测
                Roll = 0
                Rolleye = 0
                Rollmouth = 0

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw each of them
            for (i, (x, y)) in enumerate(shape):
                if i == 33:
                    # something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                    image_points[0] = np.array([x, y], dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 8:
                    # something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                    image_points[1] = np.array([x, y], dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 36:
                    # something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                    image_points[2] = np.array([x, y], dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 45:
                    # something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                    image_points[3] = np.array([x, y], dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 48:
                    # something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                    image_points[4] = np.array([x, y], dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 54:
                    # something to our key landmarks
                    # save to our new key point list
                    # i.e. keypoints = [(i,(x,y))]
                    image_points[5] = np.array([x, y], dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                else:
                    # everything to all other landmarks
                    # write on frame in Red
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # Draw the determinant image points onto the person's face
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            (head_tilt_degree, start_point, end_point,
             end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
            cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

            if head_tilt_degree:
                cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # extract the mouth coordinates, then use the
            # coordinates to compute the mouth aspect ratio
        # show the frameq
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # print(image_points)
        # do a bit of cleanup
        # cv2.destroyAllWindows()
        # vs.stop()


        global outputFrame,lock

        with lock:
            outputFrame = frame.copy()
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True, use_reloader=False)

