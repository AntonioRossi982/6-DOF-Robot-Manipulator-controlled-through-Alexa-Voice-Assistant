import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory, get_package_prefix

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# def generate_launch_description():
#     arduinobot_description = get_package_share_directory('arduinobot_description')
#     arduinobot_description_share = get_package_prefix('arduinobot_description')
#     gazebo_ros_dir = get_package_share_directory('gazebo_ros')

#     model_arg = DeclareLaunchArgument(name='model', default_value=os.path.join(
#                                         arduinobot_description, 'urdf', 'arduinobot.urdf.xacro'
#                                         ),
#                                       description='Absolute path to robot urdf file'
#     )

#     model_path = os.path.join(arduinobot_description, "models")
#     model_path += pathsep + os.path.join(arduinobot_description_share, "share")

#     env_var = SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path)

#     robot_description = ParameterValue(Command(['xacro ', LaunchConfiguration('model')]),
#                                        value_type=str)

#     robot_state_publisher_node = Node(
#         package='robot_state_publisher',
#         executable='robot_state_publisher',
#         parameters=[{'robot_description': robot_description}]
#     )

#     start_gazebo_server = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')
#         )
#     )

#     start_gazebo_client = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(gazebo_ros_dir, 'launch', 'gzclient.launch.py')
#         )
#     )

#     spawn_robot = Node(package='gazebo_ros', executable='spawn_entity.py',
#                         arguments=['-entity', 'arduinobot',
#                                    '-topic', 'robot_description',
#                                   ],
#                         output='screen'
#     )

#     return LaunchDescription([
#         env_var,
#         model_arg,
#         start_gazebo_server,
#         start_gazebo_client,
#         robot_state_publisher_node,
#         spawn_robot
#     ])

def generate_launch_description():
    arduinobot_description = get_package_share_directory("arduinobot_description")

    model_arg = DeclareLaunchArgument(name="model", default_value=os.path.join(
                                        arduinobot_description, "urdf", "arduinobot.urdf.xacro"
                                        ),
                                      description="Absolute path to robot urdf file"
    )

    gazebo_resource_path = SetEnvironmentVariable(
        # name="GZ_SIM_RESOURCE_PATH",
        name ="IGN_GAZEBO_RESOURCE_PATH",
        value=[
            str(Path(arduinobot_description).parent.resolve())
            ]
        )
    
    ros_distro = os.environ["ROS_DISTRO"]
    is_ignition = "True" if ros_distro == "humble" else "False"

    robot_description = ParameterValue(Command([
            "xacro ",
            LaunchConfiguration("model"),
            " is_ignition:=",
            is_ignition
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": True}]
    )

    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory("ros_gz_sim"), "launch"), "/gz_sim.launch.py"]),
                launch_arguments=[
                    ("gz_args", [" -v 3 -r empty.sdf"] # Verbose settled to v 3 instead v 4 
                    )
                ]
             )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description",
                   "-name", "arduinobot"],
    )

    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ]
    )

    return LaunchDescription([
        model_arg,
        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge
    ])