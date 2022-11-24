cmake_minimum_required(VERSION 3.0.2)
project(yolact_edge_ros)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
)

catkin_package(
  DEPENDS rospy sensor_msgs cv_bridge
)

include_directories(
${catkin_INCLUDE_DIRS}
)


catkin_install_python(PROGRAMS
  scripts/yolact_edge_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
