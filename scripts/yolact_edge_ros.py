#!/usr/bin/env python

import sys

yolact_path = '/root/yolact_edge'
sys.path.append(yolact_path)
yolact_path = '/root/ai_utils'
sys.path.append(yolact_path)

from ai_utils.detectors.YolactEdgeInference import YolactEdgeInference
import numpy as np
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

class YolactEdgeRos:

    def __init__(self, input_camera_name, output_camera_name, yolact_edge_weights, score_threshold) -> None:
        self.bridge = CvBridge()
        self.yolact_edge = YolactEdgeInference(model_weights=yolact_edge_weights, score_threshold=score_threshold, return_img=True, display_img=False)

        self.camera_sub = rospy.Subscriber(input_camera_name, Image, self.yolact_callback)
        self.camera_pub = rospy.Publisher(output_camera_name, Image)


    def yolact_callback(self, data):
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        img = np.array(img)
        _, yolact_img = self.yolact_edge.img_inference(img)
        self.camera_pub.publish(self.bridge.cv2_to_imgmsg(yolact_img))
        

if __name__ == '__main__':
    
    rospy.init_node('yolact_edge_ros', anonymous=True)

    input_camera_name = rospy.get_param("~input_camera_name", "")
    output_camera_name = rospy.get_param("~output_camera_name", "")
    yolact_edge_weights = rospy.get_param("~yolact_edge_weights" ,"/root/yolact_edge/weights/yolact_edge_resnet50_54_800000.pth")
    score_threshold =  rospy.get_param("~score_threshold", 0.6)

    if input_camera_name == "" or output_camera_name=="":
        rospy.logerr("input_camera_name or output_camera_name cannot be an empty string.")
        exit(1)

    yolact_edge = YolactEdgeRos(input_camera_name, output_camera_name, yolact_edge_weights, score_threshold)
    
    rospy.loginfo("Subscriber initialized.")
    rospy.spin()