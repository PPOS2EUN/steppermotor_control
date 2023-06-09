# import sys
# import argparse
# import Jetson.GPIO as GPIO
# from time import sleep
# import cv2
# import time
#
# from jetson_inference import detectNet
# from jetson_utils import videoSource, videoOutput, Log
#
# class Boeun:
#     def __init__(self, net, input, output):
#         self.motor = 1
#         self.dir = GPIO.HIGH
#         self.centerY = 0
#         self.net = net
#         self.input = input
#         self.output = output
#         self.DIR = 10
#         self.STEP = 8
#         GPIO.setmode(GPIO.BCM)
#         GPIO.setup(self.DIR, GPIO.OUT)
#         GPIO.setup(self.STEP, GPIO.OUT)
#         GPIO.output(self.DIR, GPIO.HIGH)
#
#     def __change_dir(self):
#         k = self.centerY
#         print("k = ", k, end="\n")
#         if k < 339 and 0 < k:
#             self.dir = GPIO.LOW
#             return True
#         elif 379 < k and k < 719:
#             self.dir = GPIO.HIGH
#             return True
#         elif 339 <= k and k <= 379:
#             print("center")
#             return False
#         print("motor = ", self.motor, end="\n")
#
#     def __status_motor(self, motor):
#         self.motor = motor
#         if self.dir == GPIO.HIGH:
#             self.motor += 1
#         elif self.dir == GPIO.LOW:
#             self.motor -= 1
#         print(self.motor)
#
#     def run(self):
#         detections, roiArea, roiCenter = self.face_detection()
#         if len(detections) > 0:
#             cur_center = roiCenter[roiArea.index(max(roiArea))]
#             print("cur_center", cur_center)
#             self.centerY = cur_center
#             GPIO.output(self.DIR, self.dir)
#             if self.__change_dir():
#                 self.__status_motor(self.motor)
#                 GPIO.output(self.STEP, GPIO.HIGH)
#                 sleep(.005)
#                 GPIO.output(self.STEP, GPIO.LOW)
#
#     def face_detection(self):
#         frame = self.input.Capture()
#         frameFace, bboxes = self.getFaceBox(frame)
#         detections = self.net.Detect(frameFace, overlay=args.overlay)
#         roiArea = []
#         roiCenter = []
#         for detection in detections:
#             roiArea.append(detection.Area)
#             roiCenter.append(detection.Center[1])
#
#         self.output.Render(frameFace)
#         self.output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, self.net.GetNetworkFPS()))
#
#         return detections, roiArea, roiCenter
#
#     def getFaceBox(self, frame):
#         frameOpencvDnn = frame.copy()
#         frameHeight = frameOpencvDnn.shape[0]
#         frameWidth = frameOpencvDnn.shape[1]
#         blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
#
#         faceNet.setInput(blob)
#         detections = faceNet.forward()
#         bboxes = []
#
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > self.conf_threshold:
#                 x1 = int(detections[0, 0, i, 3] * frameWidth)
#                 y1 = int(detections[0, 0, i, 4] * frameHeight)
#                 x2 = int(detections[0, 0, i, 5] * frameWidth)
#                 y2 = int(detections[0, 0, i, 6] * frameHeight)
#                 bboxes.append([x1, y1, x2, y2])
#                 cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
#
#         return frameOpencvDnn, bboxes
#
#
# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"
#
# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"
#
# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"
#
# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']
#
# ageNet = cv2.dnn.readNet(ageModel, ageProto)
# genderNet = cv2.dnn.readNet(genderModel, genderProto)
# faceNet = cv2.dnn.readNet(faceModel, faceProto)
#
# cap = cv2.VideoCapture(0)
# padding = 20
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="Locate objects in a live camera stream using an object detection DNN.",
#         formatter_class=argparse.RawTextHelpFormatter,
#         epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())
#
#     parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
#     parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
#     parser.add_argument("—network", type=str, default="facedetect",
#                         help="pre-trained model to load (see below for options)")
#     parser.add_argument("—overlay", type=str, default="box,labels,conf",
#                         help="detection overlay flags (e.g. —overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
#     parser.add_argument("—threshold", type=float, default=0.5, help="minimum detection threshold to use")
#
#     is_headless = ["—headless"] if sys.argv[0].find('console.py') != -1 else [""]
#
#     try:
#         args = parser.parse_known_args()[0]
#     except:
#         print("")
#         parser.print_help()
#         sys.exit(0)
#     args.input = '/dev/video0'
#     args.network = 'facedetect'
#     args.overlay = 'box,labels,conf'
#     args.threshold = 0.5
#
#     input = videoSource(args.input, argv=sys.argv)
#     output = videoOutput(args.output, argv=sys.argv + is_headless)
#
#     net = detectNet(args.network, sys.argv, args.threshold)
#
#     boeun = Boeun(net, input, output)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         boeun.run()
#         t = time.time()
#
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#
# cap.release()
# cv2.destroyAllWindows()

import math
import time
import argparse
from jetson_utils import videoSource
import Jetson.GPIO as GPIO
from time import sleep
import numpy as np

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log


class Capstone:

    def __init__(self, ageModel, ageProto, genderModel, genderProto, faceModel, faceProto):
        self.ageNet = detectNet(ageModel, ageProto)
        self.genderNet = detectNet(genderModel, genderProto)
        self.faceNet = detectNet(faceModel, faceProto)
        self.cap = videoSource("csi://0", argv=["--input-flip=rotate-180"])
        self.padding = 20
        self.genderList = ['Male', 'Female']
        self.ageList = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
        self.motor = 1
        self.dir = GPIO.HIGH
        self.centerY = 0
        self.DIR = 10
        self.STEP = 8
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.DIR, GPIO.OUT)
        GPIO.setup(self.STEP, GPIO.OUT)
        GPIO.output(self.DIR, GPIO.HIGH)

    def getFaceBox(self, frame, conf_threshold=0.75):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = self.faceNet.cuda().SetInputBuffer(0, frameOpencvDnn)

        self.faceNet.Detect()
        detections = self.faceNet.GetDetections()
        bboxes = []

        for detection in detections:
            confidence = detection.Confidence
            if confidence > conf_threshold:
                x1 = int(detection.Left)
                y1 = int(detection.Top)
                x2 = int(detection.Right)
                y2 = int(detection.Bottom)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

        return frameOpencvDnn, bboxes

    def __change_dir(self):
        k = self.centerY
        print("k =", k, end="\n")
        if k < 339 and 0 < k:
            self.dir = GPIO.LOW
            return True
        elif 379 < k and k < 719:
            self.dir = GPIO.HIGH
            return True
        elif 339 <= k and k <= 379:
            print("center")
            return False
        print("motor =", self.motor, end="\n")

    def __status_motor(self, motor):
        self.motor = motor
        if self.dir == GPIO.HIGH:
            self.motor += 1
        elif self.dir == GPIO.LOW:
            self.motor -= 1
        print(self.motor)

    def face_detection(self):
        img = self.cap.Capture()
        detections = self.faceNet.Detect(img)
        print("detected {:d} objects in image".format(len(detections)))
        roiArea = []
        roiCenter = []
        for detection in detections:
            roiArea.append(detection.Area)
            roiCenter.append(detection.Center[1])

        return detections, roiArea, roiCenter

    def run(self):
        detections, roiArea, roiCenter = self.face_detection()
        if len(detections) > 0:
            cur_center = roiCenter[roiArea.index(max(roiArea))]
            print("cur_center", cur_center)
            self.centerY = cur_center
            GPIO.output(self.DIR, self.dir)
            if self.__change_dir():
                self.__status_motor(self.motor)
                GPIO.output(self.STEP, GPIO.HIGH)
                sleep(.005)
                GPIO.output(self.STEP, GPIO.LOW)

            frame = self.cap.Capture()
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frameFace, bboxes = self.getFaceBox(small_frame)

            if not bboxes:
                print("No face Detected, Checking next frame")
                return

            for bbox in bboxes:
                face = small_frame[
                    max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, frame.shape[0] - 1),
                    max(0, bbox[0] - self.padding):min(bbox[2] + self.padding, frame.shape[1] - 1)]
                blob = self.faceNet.cuda().SetInputBuffer(0, face)

                self.genderNet.Detect()
                genderPreds = self.genderNet.GetDetections()
                gender = genderList[genderPreds[0].ClassID]
                print("Gender: {}, conf = {:.3f}".format(gender, genderPreds[0].Confidence))

                self.ageNet.Detect()
                agePreds = self.ageNet.GetDetections()
                age = ageList[agePreds[0].ClassID]
                print("Age Output: {}".format(agePreds))
                print("Age: {}, conf = {:.3f}".format(age, agePreds[0].Confidence))

                label = "{},{}".format(gender, age)
                cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Age Gender Demo", frameFace)

    def cleanup(self):
        GPIO.cleanup()


# Set the paths to your age and gender models
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

capstone = Capstone(ageModel, ageProto, genderModel, genderProto, faceModel, faceProto)

try:
    while True:
        capstone.run()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

capstone.cleanup()
