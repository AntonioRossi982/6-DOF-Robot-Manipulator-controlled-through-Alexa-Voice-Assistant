from setuptools import find_packages, setup

package_name = 'mujoco_ros2_bridge'

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
    maintainer_email='antonio.rossi9829@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mujoco_sim = mujoco_ros2_bridge.mujoco_sim_node:main',
            'pd_controller = mujoco_ros2_bridge.pd_controller_node:main',
            'mission_planner = mujoco_ros2_bridge.mission_planner_node:main',
        ],
    },
)
