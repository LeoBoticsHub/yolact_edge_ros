#!/usr/bin/env python
'''----------------------------------------------------------------------------------------------------------------------------------
# Copyright (C) 2022
#
# author: Federico Rollo
# mail: rollo.f96@gmail.com
#
# Institute: Leonardo Labs (Leonardo S.p.a - Istituto Italiano di tecnologia)
#
# This file is part of yolact_edge_ros. <https://github.com/IASRobolab/yolact_edge_ros>
#
# yolact_edge_ros is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# yolact_edge_ros is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License. If not, see http://www.gnu.org/licenses/
---------------------------------------------------------------------------------------------------------------------------------'''

from ai_utils.detectors.YolactEdgeInference import YolactEdgeInference
from camera_utils.cameras.IntelRealsense import IntelRealsense

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

        

if __name__ == '__main__':
    
    rospy.init_node('yolact_edge_ros', anonymous=True)

    bridge = CvBridge()

    disable_tensorrt = rospy.get_param("~disable_tensorrt", False)
    output_camera_name = rospy.get_param("~output_camera_name", "")
    yolact_edge_weights = rospy.get_param("~yolact_edge_weights" ,"/root/yolact_edge/weights/yolact_edge_resnet50_54_800000.pth")
    score_threshold =  rospy.get_param("~score_threshold", 0.6)

    use_intel_camera =  rospy.get_param("~use_intel_camera", False)

    if output_camera_name=="":
        rospy.logerr("input_camera_name or output_camera_name cannot be an empty string.")
        exit(1)

    yolact_edge = YolactEdgeInference(disable_tensorrt=disable_tensorrt, model_weights=yolact_edge_weights, score_threshold=score_threshold, return_img=True, display_img=False)
    camera_pub = rospy.Publisher(output_camera_name, Image, queue_size=100)

    camera = IntelRealsense(IntelRealsense.Resolution.LOW)

    rospy.loginfo("Starting YolactEdgeRos.")

    while not rospy.is_shutdown():
        img = camera.get_rgb()
        _, yolact_img = yolact_edge.img_inference(img)
        camera_pub.publish(bridge.cv2_to_imgmsg(yolact_img))

