#!/usr/bin/env python
import os
import tempfile
import threading
from six.moves import urllib

import PIL
import numpy as np
import cv2
import importlib
import time
from argparse import ArgumentParser

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from erfnet_ros.erfnet import ERFNet
from erfnet_ros.transform import Relabel, ToLabel, Colorize

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class TestingNode(object):
    def __init__(self):
        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 100)  #default=100Hz
        camera_info_topic = rospy.get_param("~camera_info_topic", '/pylon_camera_node/camera_info')
        rospy.Subscriber(camera_info_topic, CameraInfo, self._camera_info_callback, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/erfnet_ros/camera_info', CameraInfo, queue_size=1)

    def _camera_info_callback(self, msg):
        rospy.logdebug("Received camera info.")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


    def run(self):
        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():
            # mutel lock mechanism
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                # Run detection
                P_arr = np.asarray(msg.P)
                P_arr = P_arr / 3
                P_arr[-2] = 1
                P = P_arr.tolist()
                camera_info_msg = msg
                camera_info_msg.P = P
                self.camera_info_pub.publish(camera_info_msg)
                #self.img_pub.publish(img_msg)

            #if self._visualize:
                # Overlay Semantic mask on RGB Image
            rate.sleep()


def main():
    rospy.init_node("testing_node")
    node = TestingNode()
    print("[DEBUG] Node Initalization done ~")
    node.run()

if __name__ == '__main__':
    main()
