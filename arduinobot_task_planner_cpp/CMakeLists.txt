cmake_minimum_required(VERSION 3.8)
project(arduinobot_task_planner_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(control_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)

add_executable(collision_monitor src/collision_monitor.cpp)
ament_target_dependencies(collision_monitor rclcpp std_msgs sensor_msgs)

add_executable(robot_interface_test src/robot_interface_test.cpp)
ament_target_dependencies(robot_interface_test rclcpp rclcpp_action control_msgs trajectory_msgs builtin_interfaces)

add_executable(robot_interface src/robot_interface.cpp)
ament_target_dependencies(robot_interface rclcpp rclcpp_action control_msgs trajectory_msgs builtin_interfaces)

add_executable(scene_manager src/scene_manager.cpp)
ament_target_dependencies(scene_manager rclcpp moveit_msgs shape_msgs geometry_msgs)

add_executable(task_planner src/task_planner.cpp)
ament_target_dependencies(task_planner rclcpp rclcpp_action control_msgs trajectory_msgs builtin_interfaces)

install(TARGETS
  collision_monitor
  robot_interface
  robot_interface_test
  scene_manager
  task_planner
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
