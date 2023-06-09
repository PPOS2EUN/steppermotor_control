#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import Jetson.GPIO as GPIO
from time import sleep
import numpy as np

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="/dev/video0", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="facedetect",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf",
                    help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)
#print(args)
args.input = '/dev/video0'
args.network = 'facedetect'
args.overlay = 'box,labels,conf'
args.threshold = 0.5
#print(args,is_headless)

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv + is_headless)


# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

#print(sys.argv)
exit()

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = detectNet(model="model/ssd-mobilenet.onnx", labels="model/labels.txt",
#                 input_blob="input_0", output_cvg="scores", output_bbox="boxes",
#                 threshold=args.threshold)

# process frames until EOS or the user exits

global DIR, STEP

DIR = 10
STEP = 8

GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)
GPIO.output(DIR, GPIO.HIGH)


class Set:
    def __init__(self):
        self.motor = 1
        self.centerY = 0
        self.roiArea = np.array([])
        self.roiCenter = np.array([])
        self.face_detection(self.centerY, self.roiArea, self.roiCenter)
        self.dir = GPIO.HIGH
        status = self.moving_motor()
        self.change_dir()
        GPIO.output(DIR, self.dir)
        if status:
            self.status_motor(self.motor)
            GPIO.output(STEP, GPIO.HIGH)
            sleep(.005)
            GPIO.output(STEP, GPIO.LOW)
    def setDIR(self, dir):
        self.dir = dir
        return self.dir

    # centerY값 좌표에 따라서 상하 움직임 바꾸는 함수
    def change_dir(self):
        k = self.centerY
        print("k = ", k, end="\n")
        if k < 349 and 0 < k:
            self.dir = GPIO.LOW
            return True
        elif 369 < k and k < 719:
            self.dir = GPIO.HIGH
            return True
        elif 349 <= k and k <= 369:
            print("center")
            return False
        print("motor = ", self.motor, end="\n")

    # 0 또는 10666이 되면 모터 작동 멈추게 하는 함수. return값이 False면 작동중지.
    def moving_motor(self):
        if self.motor == 0:
            return False
        elif self.motor == 10666:
            return False
        else:
            return True

    # 모터의 현재 스텝 개수 파악
    def status_motor(self, motor):
        self.motor = motor
        if self.dir is True:
            self.motor += 1
        elif self.dir is False:
            self.motor -= 1
        print(self.motor)

    # 얼굴 감지 및 가장 큰 영역 판별 후 센터값을 잡는 함수
    def face_detection(self, centerY, roiArea, roiCenter):
        # capture the next image
        img = input.Capture()
        # detect objects in the image (with overlay)
        detections = net.Detect(img, overlay=args.overlay)
        # print(args.overlay)
        print("detected {:d} objects in image".format(len(detections)))

        for detection in detections:
            print(detection)
            print(detection.Center)
            self.roiArea = np.array(detection.Area)
            self.roiCenter = np.array(detection.Center)
            # classID -> list detection 위치 번호
            # Area -> height x width
            # center -> detection의 center값
            print("classID = ", detection.ClassID)
            print("center x = ", detection.Center[0])
            print("center y = ", detection.Center[1])

        #detection한 객체의 가장 큰 영역을 찾아서 그 영역의 센터값에 맞추게 하기.
        maxIndex = np.argmax(self.roiArea)
        print(maxIndex)
        #self.centerY = self.roiCenter[maxIndex][1]

        print("현재 가장 큰 값 : ", maxIndex, "번, ", self.centerY)
        output.Render(img)
        output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

if __name__ == '__main__':
    while True:
        Set()

        if not input.IsStreaming() or not output.IsStreaming():
            GPIO.cleanup()
            break
