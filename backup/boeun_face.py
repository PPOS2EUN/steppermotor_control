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

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
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

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv + is_headless)

# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

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


class step:
    def __init__(self):
        self.id = 0
        self.motor = 0
        self.centerY = 0
        self.face_detection(self.id, self.centerY)
        self.SetStepDirection(self.motor)

    def face_detection(self, id, centerY):
        # capture the next image
        img = input.Capture()
        self.id = id
        self.centerY = centerY
        # detect objects in the image (with overlay)
        detections = net.Detect(img, overlay=args.overlay)
        # print(args.overlay)
        print("detected {:d} objects in image".format(len(detections)))

        for detection in detections:
            print(detection)
            print(detection.Center)
            self.id = detection.ClassID
            self.centerY = detection.Center[1]
            print("center x = ", detection.Center[0])
            print("center y = ", detection.Center[1])

        output.Render(img)
        output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    def SetStepDirection(self, motor):
        self.motor = motor
        print(self.centerY, self.id)
        print("current motor step:", end=" ")
        print(self.motor)

        if self.centerY <= 349 and 369 <= self.centerY:
            print("============center============")
            return "center"
        # UP
        elif 0 < self.centerY and self.centerY < 349:
            GPIO.output(DIR, GPIO.HIGH)
            self.motor = self.motor + 1
            return "CW"
        # DOWN
        elif 369 < self.centerY and self.centerY < 719:
            GPIO.output(DIR, GPIO.LOW)
            self.motor = self.motor - 1
            return "CCW"
        # HOME
        elif self.motor == 0 or self.motor == 10666:
            return "HOME"


if __name__ == '__main__':

    while True:
        s = step()
        if not s.SetStepDirection(0) == "CW" or not s.SetStepDirection(0) == "CCW":
            continue
        GPIO.output(STEP, GPIO.HIGH)
        sleep(.005)
        GPIO.output(STEP, GPIO.LOW)

        if not input.IsStreaming() or not output.IsStreaming():
            GPIO.cleanup()
            break

