cmake_minimum_required(VERSION 3.0.2)
project(yolact_edge_ros)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
)

catkin_install_python(PROGRAMS
  scripts/yolact_edge_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
