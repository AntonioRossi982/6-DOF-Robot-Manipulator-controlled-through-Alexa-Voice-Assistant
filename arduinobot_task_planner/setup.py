from setuptools import find_packages, setup

package_name = 'arduinobot_task_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='antonio',
    maintainer_email='antonio@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'task_planner = arduinobot_task_planner.task_planner:main',
            'scene_manager = arduinobot_task_planner.scene_manager:main',
            'collision_monitor = arduinobot_task_planner.collision_monitor:main',
            'robot_interface_test = arduinobot_task_planner.robot_interface_test:main',
            'robot_interface = arduinobot_task_planner.robot_interface:main',
        ],
    },
)
