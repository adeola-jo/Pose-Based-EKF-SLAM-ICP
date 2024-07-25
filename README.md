# Pose-Based EKF SLAM using ICP Scan-Matching on a Kobuki Turtlebot
This package houses all the packages, scripts, etc., for the Hands-On Localization project on 
Pose-Based EKF SLAM using ICP Scan-Matching

## Members of the goup:

This project has been carried out by:

Moses Chuka Ebere

Joseph Oloruntoba Adeola

Preeti Verma

## Requirements
- ROS version: Noetic
- Operating System: Ubuntu 20.04 
- [Stonefish](https://github.com/patrykcieslak/stonefish) Library 
- Other required packages: Stonefish ros, kobuki_description, swiftpro_description, turtlebot_description, octomap_server, python shapely library, python scikit-learn library, python open3d library

## Installation
1. Install/clone the required dependencies:

```
git clone https://github.com/patrykcieslak/stonefish_ros.git
git clone https://bitbucket.com/udg_cirs/kobuki_description.git # Mobile base
git clone https://bitbucket.com/udg_cirs/swiftpro_description.git # Manipulator
git clone https://bitbucket.com/udg_cirs/turtlebot_description.git # Mobile base + manipulator (whole robot)
sudo apt install ros-noetic-realsense2-description # (Realsense camera)
git clone https://bitbucket.org/udg_cirs/turtlebot_simulation.git
sudo apt-get install ros-<distro>-octomap-msgs ros-<distro>-octomap-ros
```
```
pip install open3d

pip install shapely

pip install scikit-learn

```

2. Clone/download the hands_on_localization package into your catkin workspace.

3. Build the packages:

```
cd ~/catkin_ws
catkin_make or catkin build (depending on how the catkin workspace was initially built)
```

5. Source your catkin workspace:

```
source ~/catkin_ws/devel/setup.bash
```

## Usage/Execution
This package comes with two launch files: `rviz_slam.launch` and `octomap_slam.launch`. rviz_slam.launch launches the kobuki base in rviz only while octomap_slam.launch launches the kobuki base in rviz and octomap.
Additionally, these launch files are already configured to access the necessary dependencies from turtlebot_simulation and kobuki_description. There are two python scripts in the src folder of this project,

1. slam_node.py: runs generic the icp pose_based  and publishes several topics for visual information in rviz
2. slam_with_plots.py: Executes the slam algorithm and generates plots into slam_results folder that is 
    automatically generated.
3. slam_node_fast.py: runs the slam algorithm publishing only the robots belief


To launch all the setup, use any of the following commands (per your requirement):

```
roslaunch hands_on_localization <...>.launch
```
Replace <...> with the launch file of your choosing. 

Now, run the project node, `slam_node.py`.

```
rosrun hands_on_localization slam_node.py
```

To manually control the robot using teleop

```
rosrun hands_on_localization teleop_node.py
```

Kindly follow the instruction printed on the teleop_node screen to control the robot.




## Resources
Some resources are contained in root folder of this package.
1. config: contains rviz configuration
2. launch: The necessary launch files are contained here.
3. scenarios: Some required scenario files are contained here. 
4. src: All python scripts


## Rviz Resuls Color Code:

**Purple & Red laserscans:** Overlapping scans

**White laser scan:**: Map of all aligned scans

**Black:** Cloned poses

**Yellow:** Robot's belief trajectory

**Green path:** Ground truth trajectory

## Troubleshooting
If you encounter issues while using the files in this package, 
- Ensure you build the package using the right method. 
- Ensure that you run 'rospack profile' once the package is built. 
- Ideally, you should source the workspace in which you place this package when you want to use it. 
- Make sure all required dependencies are installed and up-to-date.
- Check the ROS logs for error messages or other relevant information.
- Consult the package documentation or search online forums for help.

---

<sup>
Moses Chuka Ebere, Joseph Oloruntoba Adeola, & Preeti Verma - 
April 2023.
</sup>
