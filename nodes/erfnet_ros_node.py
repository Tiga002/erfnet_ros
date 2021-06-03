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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ERFNetNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 100)  #default=100Hz
        self._visualize = rospy.get_param('~visualize', True)

        rgb_input_topic = rospy.get_param('~rgb_input_topic', '/pylon_camera_node/image_rect_color')
        rospy.Subscriber(rgb_input_topic, Image, self._image_callback, queue_size=1)

        self.mask_pub = rospy.Publisher('/semantic_mask', Image, queue_size=1)
        self.vis_pub = rospy.Publisher('/semantic_mask_viz', Image, queue_size=1)

        num_of_class = rospy.get_param('~num_of_class', 20)
        load_dir = rospy.get_param('~load_dir', '/home/tiga/Documents/IRP/dev/erfnet_pytorch/save/erfnet_training_remote_v3/')
        weights_name = rospy.get_param('~weights_dir', 'model_best.pth')
        weights_dir = load_dir + weights_name
        if(not os.path.exists(load_dir)):
            print ("Error: Directory could not be loaded")

        self._model = ERFNet(num_of_class)
        self._model = torch.nn.DataParallel(self._model)
        self._model = self._model.cuda()  #gpu execution

        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        self._model = load_my_state_dict(self._model, torch.load(weights_dir))
        # Set model to Evaluation mode
        self._model.eval()
        print ("Model and weights LOADED successfully")

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
                mask = self.detect(msg)
                rospy.logdebug("Publishing semantic labels.")
                # CV Image -> ROS Image msg
                mask_msg = self._cv_bridge.cv2_to_imgmsg(mask, 'bgr8')
                mask_msg.header = msg.header
                # Publish the semantic mask
                self.mask_pub.publish(mask_msg)

            #if self._visualize:
                # Overlay Semantic mask on RGB Image
            rate.sleep()


    def detect(self, msg):
        # Convert the ROS image msg into cv image(bgr)
        cv_image = self._cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        # BGR image -> RGB image
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # CV Image -> PIL Image
        pil_img = PIL.Image.fromarray(cv_image)
        # Resize to 640X360
        pil_img = Resize((360,640), PIL.Image.BILINEAR)(pil_img)
        # PIL Image -> Normalize -> Tensor
        image_transforms = Compose([
            ToTensor(),
            Normalize((0.496588, 0.59493099, 0.53358843), (0.496588, 0.59493099, 0.53358843))])
        img = image_transforms(pil_img)
        img = img.resize_((1,3,360,640))
        #print("[DEBUG] input size = {0}".format(img.shape))

        # Run Semantic Segmentation
        img = img.cuda()
        inputs = Variable(img)
        with torch.no_grad():
            outputs = self._model(inputs)

        label = outputs[0].max(0)[1].byte().cpu().data
        label_color = Colorize()(label.unsqueeze(0))

        # Tensor -> PIL Image
        mask = ToPILImage()(label_color)
        # PIL Image -> cv image
        mask = np.array(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        return mask

    def _image_callback(self, msg):
        rospy.logdebug("Got an image.")

        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

def main():
    rospy.init_node("erfnet_ros_node")
    node = ERFNetNode()
    print("[DEBUG] Node Initalization done ~")
    node.run()

if __name__ == '__main__':
    main()
