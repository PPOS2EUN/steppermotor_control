#!/usr/bin/env python3

import sys
import argparse
import Jetson.GPIO as GPIO
from time import sleep
import numpy as np

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log



# note: to hard-code the paths to load a model, the following API can be used:
#
# net = detectNet(model="model/ssd-mobilenet.onnx", labels="model/labels.txt",
#                 input_blob="input_0", output_cvg="scores", output_bbox="boxes",
#                 threshold=args.threshold)

# process frames until EOS or the user exits


GPIO.setmode(GPIO.BCM)

class Boeun:
    def __init__(self,net,input,output):
        self.motor = 1
        self.dir = GPIO.HIGH
        self.centerY = 0
        self.net = net
        self.input = input
        self.output = output
        self.DIR = 10
        self.STEP = 8
        GPIO.setup(self.DIR, GPIO.OUT)
        GPIO.setup(self.STEP, GPIO.OUT)
        GPIO.output(self.DIR, GPIO.HIGH)


    # note centerY값 좌표에 따라서 상하 움직임 바꾸는 함수
    def __change_dir(self):
        k = self.centerY + 20
        print("k = ", k, end="\n")
        if k < 319 and 0 < k:
            self.dir = GPIO.LOW
            return True
        elif 379 < k and k < 719:
            self.dir = GPIO.HIGH
            return True
        elif 339 <= k and k <= 379:
            print("center")
            return False
        print("motor = ", self.motor, end="\n")

    # TODO display center setting, display height / 2 HAMSU here

    # note 모터의 현재 스텝 개수 파악
    def __status_motor(self, motor):
        self.motor = motor
        if self.dir == GPIO.HIGH:
            self.motor += 1
        elif self.dir == GPIO.LOW:
            self.motor -= 1
        print(self.motor)

    # 얼굴 감지 및 가장 큰 영역 판별 후 센터값을 잡는 함수
    def run(self):
        detections, roiArea, roiCenter = self.face_detection()
        if (len(detections) > 0):
            cur_center = roiCenter[roiArea.index(max(roiArea))]
            print("cur_center", cur_center)
            self.centerY = cur_center
            GPIO.output(self.DIR, self.dir)
            if self.__change_dir():
                # self.__status_motor(self.motor)
                GPIO.output(self.STEP, GPIO.HIGH)
                sleep(.005)
                GPIO.output(self.STEP, GPIO.LOW)

    def face_detection(self):
        # capture the next image
        img = self.input.Capture()
        # detect objects in the image (with overlay)
        detections = self.net.Detect(img, overlay=args.overlay)
        # print(args.overlay)
        print("detected {:d} objects in image".format(len(detections)))
        roiArea = []
        roiCenter = []
        for detection in detections:
            roiArea.append(detection.Area)
            roiCenter.append(detection.Center[1])

        self.output.Render(img)
        self.output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))
        return detections, roiArea, roiCenter

        # note detection한 객체의 가장 큰 영역을 찾아서 그 영역의 센터값에 맞추게 하기.

if __name__ == '__main__':
    # parse the command line
    parser = argparse.ArgumentParser(
        description="Locate objects in a live camera stream using an object detection DNN.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
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
    args.input = '/dev/video0'
    args.network = 'facedetect'
    args.overlay = 'box,labels,conf'
    args.threshold = 0.5

    # create video sources and outputs
    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv + is_headless)

    # load the object detection network
    net = detectNet(args.network, sys.argv, args.threshold)

    boeun = Boeun(net,input,output)

    while True:
        boeun.run()
        if not input.IsStreaming() or not output.IsStreaming():
            GPIO.cleanup()
            break
