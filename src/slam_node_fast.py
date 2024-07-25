#! /usr/bin/env python3


import sensor_msgs.point_cloud2 as ab
from sklearn.neighbors import NearestNeighbors
import threading
import rospy
import tf
import numpy as np
from sensor_msgs.msg import JointState, PointCloud2, LaserScan, Imu
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry, Path
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from shapely.geometry.polygon import Polygon, Point
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA,  Header
import threading
from laser_scan_to_point_cloud import LaserProjection


class PoseBasedSlam(object):
    """
    A class to implement the pose based slam algorithm
    """

    def __init__(self) -> None:

        #initialize the robot's pose to zero. the pose is a vector of 3 elements: x, y and theta
        
        self.xk = np.zeros((3, 1))
        # Edit the initial position of the robot
        self.xk[0,0] = 3.0
        self.xk[1,0] = -0.78
        self.xk[2,0] = np.pi/2.0
        self.linear_velocity = 0
        self.angular_velocity = 0


        self.xk_slam = self.xk
        
        # transformation matrix from the robot frame to the lidar frame, (x, y, z, theta)
        self.robotTlidar = np.array([[-0.02], [0], [-0.179], [0]])

        # the wheel radius and wheel distance are assumed. Not the exact values for the turtlebot
        self.wheel_radius = 0.035
        self.wheel_distance = 0.230

        # global variables declaration for the encoder velocity readings 
        self.left_wheel_ang_vel = 0
        self.right_wheel_ang_vel = 0

        # global variables declaration for the time stamps
        self.time_stamp_left = 0
        self.time_stamp_right = 0
        self.prev_time_stamp = rospy.Time.now().to_sec()

        #variable to check the joint state message
        # they are both set to true once the two wheel velocities are each received as the first message
        self.left_wheel_received = False
        self.right_wheel_received = False

        # define the noise in the wheel angular velocity readings
        self.left_wheel_encoder_ang_sigma = 0.01
        self.right_wheel_encoder_ang_sigma = 0.01
        

        # define the covariance matrix of the encoder angular and linear velocities
        self.Qk = np.diag([self.left_wheel_encoder_ang_sigma**2, 
                    self.right_wheel_encoder_ang_sigma**2])
        

        # define the initial covariance matrix of observation model with respect to the noise
        self.ICP_Vk = np.eye(3)  

        # define the initial covariance matrix of observation model with respect to the displacement
        # self.ICP_Rk = np.diag([0.04, 0.04, 0.04]) 
        # self.ICP_Rk = np.diag([0.2, 0.2, 0.2]) 
        self.ICP_Rk = np.diag([0.05,0.05,0.02])

 

        # define the initial covariance matrix of observation model with respect to the compass
        # self.compass_Vk = np.eye(1)
        self.compass_Vk = np.diag([0.0001])
        
        # define the covariance matrix of the compass
        # self.compass_Rk = np.diag([0.157]) 
        self.compass_Rk = np.diag([0.01]) 

        # define the jacobian matrices 
        self.Ak = np.zeros((3, 3))
        self.Wk = np.zeros((3, 2))

        # define the initial covariance matrix of the process noise
        self.Pk = np.eye(3) * 0.1

        self.Pk_slam = self.Pk

        # declare a transformation broadcaster and an odometry to publish the pose of the robot as a transformation and an odometry message respectively
        self.robot_pose = tf.TransformBroadcaster()
        self.kobuki_pose = Odometry()
        self.kobuki_pose_transform = tf.TransformBroadcaster()
        self.Cov = Odometry()


        # define a publisher to publish the robot's belief about its pose
        self.pub = rospy.Publisher("/kobuki/differential_drive", Odometry, queue_size=1)


        #create a subscriber to the ground truth odometry sensor
        self.sub_odom = rospy.Subscriber("/kobuki/sensors/virtual_odom_sensor", Odometry, self.ground_truth_odom_callback)

        #define a publisher to publish the ground truth odometry path
        self.pub_ground_truth_odom_path = rospy.Publisher("/ground_truth_odom_path", Path, queue_size=1)

        self.pub1 = rospy.Publisher("/pose_covariance", Odometry, queue_size=10)

        self.num_scans_received = 1

        # set the rate of the node: 10Hz
        # ps: the rate cannot be defined above the node initialization
        self.rate = rospy.Rate(10)

        # some variables for publishers
        self.lp = LaserProjection()
        self.robot_path_msg = Path()
        self.cloned_pose_path_msg = Path()
        self.ground_truth_odom_path_msg = Path()
        

        # lock to handle the callback interference
        self.lock = threading.RLock()

        # flag to handle callback interference
        self.control_flag = 'x' 

        # define a publisher to publish the point cloud
        self.point_cloud1_pub = rospy.Publisher('/point_cloud1', PointCloud2, queue_size=10)
        self.point_cloud2_pub = rospy.Publisher('/point_cloud2', PointCloud2, queue_size=10)
        self.all_aligned_scans_pub = rospy.Publisher('/all_aligned_scans', PointCloud2, queue_size=10)

        # define a publisher to publish the robot's path#
        self.pub_robot_path = rospy.Publisher('/robot_traj', Path, queue_size=10)

        #define a publisher to publish the cloned pose
        self.pub_cloned_pose_marker = rospy.Publisher('/cloned_pose_marker', Marker, queue_size=1)

        #define a publisher to publish the cloned pose as a path
        # self.pub_cloned_pose_path = rospy.Publisher('/cloned_pose_path', Path, queue_size=1)
        
        #initial id of the marker of the cloned pose
        self.cloned_marker_id = 0

        # create a list to store the scans received from the lidar
        #this corresponds to the scan history
        self.map = []

        #NOTE: Main SLAM Parameters
        # minimum threshold to check if two scans overlap using shapely polygon library
        #choose the icp method
        # 1: icp using the point to point without initial guess
        # 2: icp using the point to point with initial guess
        # 3: icp using the point to plane with initial guess
        self.ICP_method = 3

        # icp minimum distance threshold for correspondence
        self.registration_threshold = 0.1


        # choose the overlap method
        # 1: using scikit-learn knn algorithm
        # 2: using shapely polygon library
        self.overlap_method = 2
    
        # shapel polygon library minimum overlap percentage threshold
        self.overlap_threshold = 70

        # knn minimum correspondence distance
        self.knn_min_dist_thresh = 0.1

        # threshold to check if certain angle or distance is reached
        # before performing update
        self.scan_angle_threshold = 0.1
        self.scan_distance_threshold = 0.1

        # maximum number of poses to be stored in the state vector
        self.max_num_poses = 11

        # the initial index of the pose to be deleted: MJP Algorithm
        self.pose_to_delete = 1

        # Mahalanobis distance threshold
        self.chi_square_min_dist_thresh = 0.2

        # Subscribe to the joint_states to extract the velocity of the wheels.
        self.subScan = rospy.Subscriber("/kobuki/sensors/rplidar", LaserScan, self.lidar_msg_callback)

        # Subscribe to the imu topic to extract the compass reading
        self.subImu = rospy.Subscriber('/kobuki/sensors/imu', Imu, self.imu_callback)
        
        # define a subscriber to joint_states topic and a publisher to the odometry topic
        self.sub_joint_state = rospy.Subscriber("/kobuki/joint_states", JointState, self.joint_states_callback)

        pass


 
    def joint_states_callback(self, msg):
        """this function is called when a message is received from the topic
            and it updates the global variables with the latest encoder readings

        Args:
            msg (JointState): the message received from the /kobuki/joint_states topic 
        """
        # self.lock.acquire()
        if msg.name[0] == 'kobuki/wheel_left_joint':
            #add the noise to the encoder reading 
            self.left_wheel_ang_vel = msg.velocity[0] 
            self.time_stamp_left = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
            self.left_wheel_received = True
            pass

        elif msg.name[0] == 'kobuki/wheel_right_joint':

            self.right_wheel_ang_vel  = msg.velocity[0] 
            self.time_stamp_right = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
            self.right_wheel_received = True

        # check if both encoder readings are received
            if self.left_wheel_received == True and self.right_wheel_received == True:
                # perform the prediction step of the kalman filter
                self.prediction(self.left_wheel_ang_vel, self.right_wheel_ang_vel, self.time_stamp_left, self.time_stamp_right)
            else:
                print("Waiting for encoder readings")
                pass

        pass
            



    def prediction(self, left_wheel_ang_vel, right_wheel_ang_vel, time_stamp_left, time_stamp_right) -> None:
        """this function calculates the pose of the robot and publish it as a transformation

        Args:
            left_wheel_ang_vel (float): the angular velocity of the left wheel
            right_wheel_ang_vel (float): the angular velocity of the right wheel
            time_stamp_left (float): the time stamp of the left wheel encoder reading
            time_stamp_right (float): the time stamp of the right wheel encoder reading

        """
        self.lock.acquire()
        # choose the maximum time stamp between the two encoder readings
        self.current_time_stamp = max(time_stamp_left, time_stamp_right)

        # compute the time difference between the current and previous encoder readings
        delta_time = abs(self.current_time_stamp - self.prev_time_stamp) 

        # compute the linear and angular velocity based on the encoder readings
        self.linear_velocity = (self.wheel_radius/2) * (left_wheel_ang_vel + right_wheel_ang_vel)
        self.angular_velocity = (self.wheel_radius/self.wheel_distance) * (left_wheel_ang_vel - right_wheel_ang_vel)

        # use the linear and angular velocities to calculate the robot's belief
        self.xk[0,0] += self.linear_velocity * np.cos(self.xk[2,0]) * delta_time
        self.xk[1,0] += self.linear_velocity * np.sin(self.xk[2,0]) * delta_time
        self.xk[2,0] += (self.angular_velocity * delta_time)

        # extract the robot's belief from the slam belief
        self.xk_slam[-3:, :] = self.xk

        # compute the jacobian of the motion model with respect to the robot's belief
        J_fx = self.jacobianFx(self.linear_velocity, self.xk[2,0], delta_time)

        #calculate the jacobian matrix F1
        F1k = self.jacobianF1k(self.xk_slam, J_fx)

        #calculate the jacobian of the motion model with respect to the noise
        J_fw = self.jacobianFw(self.xk[2,0], delta_time)

        #calculate the jacobian matrix F2
        F2k = self.jacobianF2k(self.xk_slam, J_fw)

        #calculate the covariance matrix of the slam
        self.Pk_slam = (np.dot(F1k, np.dot(self.Pk_slam, F1k.T))) + (np.dot(F2k, np.dot(self.Qk, F2k.T)))

        # extract the robot's covariance from the slam covariance
        self.Pk = self.Pk_slam[-3:, -3:]
    
        # publish the robot's prior belief as an odometry message
        self.publish_belief(self.xk, self.Pk)

        #publish the robot's slam belief as a marker array
        if self.xk_slam.shape[0] > 3:
            self.cov_publish_belief(self.xk_slam, self.Pk_slam)

        # send the robot's pose as a transformation
        self.kobuki_pose_transform.sendTransform((self.xk[0,0], self.xk[1,0], 0), 
                                                quaternion_from_euler(0, 0, self.xk[2,0]), rospy.Time.now(), 
                                                "kobuki/base_footprint", "world_ned")

        # update the previous time stamp
        self.prev_time_stamp = self.current_time_stamp

        # when you finish publshing the robot's belief, reset the checker variables to False
        self.left_wheel_received = False
        self.right_wheel_received = False

        self.lock.release()
        print("Predicting ...")

        pass



    def lidar_msg_callback(self, msg):

        # self.lock.acquire()
        self.sub_joint_state.unregister()
        self.subImu.unregister()
        self.control_flag = 'x'

        self.lock.acquire()

        #add the first scan to the map 
        if self.num_scans_received == 1:

            # Convert LaserScan to point cloud
            pc2_msg = self.lp.projectLaser(msg)

            # convert the scan point cloud to a numpy array
            new_scan  = np.array([p[:3] for p in ab.read_points(pc2_msg)])

            # add the pose the scan was taken at to the state vector
            self.add_new_pose()

            # add the scan to the map
            self.map.append(new_scan.copy())

            self.num_scans_received += 1

            self.lock.release()
            return None


        if self.xk_slam.shape[0] > 3:
            #calculate the distance between the robot's current pose and the last pose in the map
            a = self.xk_slam[-3:-1, :].reshape(1,2)
            b = self.xk_slam[-6:-4, :].reshape(1,2)
            distance = np.linalg.norm(a - b)

            if distance >= self.scan_distance_threshold or np.abs(self.xk_slam[-1,0] - self.xk_slam[-4,0]) >= self.scan_angle_threshold:
      
                # Convert LaserScan to point cloud
                pc2_msg = self.lp.projectLaser(msg)

                # convert the scan point cloud to a numpy array
                new_scan  = np.array([p[:3] for p in ab.read_points(pc2_msg)])

                # add the pose the scan was taken at to the state vector
                self.add_new_pose()

                # add the scan to the map
                self.map.append(new_scan.copy())

                # check if the number of scans is greater than the max number of scans we want to keep            
                #NOTE: Multi-Jump Pose (MJP) Deletion Algorithm 
                if self.xk_slam.shape[0] > (3*self.max_num_poses):

                    # Get the index of the pose to delete. But since each pose has three elements
                    # we need to multiply the index by 3 and get the three elements of the pose
                    indices_to_delete = [3*self.pose_to_delete, 3*self.pose_to_delete+1, 3*self.pose_to_delete+2]
                    
                    # delete the pose from the state vector
                    self.xk_slam = np.delete(self.xk_slam, indices_to_delete, axis=0)

                    # delete the pose from the covariance matrix
                    self.Pk_slam = np.delete(self.Pk_slam, indices_to_delete, axis=0)
                    self.Pk_slam = np.delete(self.Pk_slam, indices_to_delete, axis=1)

                    # delete the scan that was taken at this pose from the map
                    self.map.pop(self.pose_to_delete)

                    # calculate the index of the next pose to delete
                    self.pose_to_delete += 1
                    if self.pose_to_delete == (self.xk_slam.shape[0]/3):
                        self.pose_to_delete = 1

                # transform the scan point cloud to the world frame
                new_scan = self.transform_pcd(new_scan, self.xk)

                # get the pose at which the new scan was taken
                new_scan_pose = self.xk # same as self.xk_slam[-3:, :]

                if len(self.map) >= 1:

                    # run through the map to find the overlapping scans except the last one
                    for mapped_scan_index, mapped_scan in enumerate(self.map[:-1]):

                        #extract the pose at which this scan was taken from the slam state vector
                        # this corresponds to the three rows from 3*mapped_scan_index to 3*mapped_scan_index + 3
                        mapped_scan_pose = self.xk_slam[3*mapped_scan_index:3*mapped_scan_index + 3, :]

                        # transform the mapped scan to the world frame
                        mapped_scan = self.transform_pcd(mapped_scan, mapped_scan_pose)

                        # check if the new scan and the mapped scan overlap
                        if self.do_scans_overlap(new_scan, mapped_scan) == True:

                            # only update with the last 3 scans that overlaps with the new scan
                            if len(self.map) - mapped_scan_index > 2:
                                continue

                            # run ICP to get the displacement measurement between the two overlapping scans
                            measurement, Rk = self.ICP_registration(new_scan_pose, mapped_scan_pose, new_scan, mapped_scan)

                            self.lidar_update(mapped_scan_index, measurement, Rk)

        #switch the control flag to the IMU
        self.control_flag = 'z'

        #subscribe to the joint states and IMU topics again
        self.sub_joint_state = rospy.Subscriber("/kobuki/joint_states", JointState, self.joint_states_callback, queue_size=1)
        self.subImu = rospy.Subscriber('/kobuki/sensors/imu', Imu, self.imu_callback, queue_size=1)
        self.lock.release() 
        pass




    def imu_callback(self, msg):
        """ imu_callback is a callback function that is called when a new IMU message comes in

        :param msg: the IMU message
        :type msg: Imu

        :return: None
        :rtype: None
        """
        self.lock.acquire()
        if self.control_flag == 'z':
            self.control_flag = 'x'

            # convert the orientation message received from quaternion to euler
            quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

            # convert the orientation message received from quaternion to euler
            _, _ , yaw_measurement = euler_from_quaternion(quaternion)
                    
            # calculate the predicted measurement and the jacobian of the measurement
            num_poses = int(self.xk_slam.shape[0]/3)

            #create a row vector of zeros of size 1 x 3*num_poses
            Hk = np.zeros((1, 3*num_poses))

            #replace the last element of the row vector with 1
            Hk[0, -1] = 1

            #multiply the row vector with the state vector to get the observation
            predicted_compass_meas = Hk @ self.xk_slam + self.compass_Vk

            #compute the kalman gain
            K = self.Pk_slam @ Hk.T @ np.linalg.inv((Hk @ self.Pk_slam @ Hk.T) + (self.compass_Vk @ self.compass_Rk @ self.compass_Vk.T))

            #compute the innovation
            innovation = self.wrap_angle(yaw_measurement - predicted_compass_meas)

            #log the predicted measurement using rospy logging
            print('Predicted heading: %s', predicted_compass_meas)

            #log the actual measurement using rospy logging
            print('Actual heading: %s', yaw_measurement)

            #log the innovation using rospy logging
            print('Heading Innovation: %s', innovation)

            #update the state vector
            self.xk_slam = self.xk_slam + np.dot(K, innovation)

            #create the identity matrix        
            I = np.eye(3*num_poses)

            #update the covariance matrix
            self.Pk_slam = (I - K @ Hk) @ self.Pk_slam @ (I - K @ Hk).T

            #extract the robot state from the state vector
            self.xk = self.xk_slam[-3:, :]

            #extract the robot covariance from the covariance matrix
            self.Pk = self.Pk_slam[-3:, -3:]

            #publish the belief
            self.publish_belief(self.xk, self.Pk)

        self.lock.release()
        pass



    def displacement_guess(self, xk, mapped_scan_pose):
        """
        This function computes the displacement between the current pose and the pose at which the scan was taken

        :param xk: the current pose
        :type xk: numpy array of shape (3,1)
        :param mapped_scan_pose: the scan to be registered
        :type mapped_scan_pose: numpy array of shape (3,n)
        :return: the displacement between the current pose and the pose at which the scan was taken
        :rtype: numpy array of shape (3,1)
        """
        
        # replace these transformations with the transformation from which the scans were taken
        invertedTransform = np.array([[np.cos(xk[2,0]), np.sin(xk[2,0]), 0, -xk[0,0]*np.cos(xk[2,0]) - xk[1,0]*np.sin(xk[2,0])], 
                       [-np.sin(xk[2,0]), np.cos(xk[2,0]), 0, xk[0,0]*np.sin(xk[2,0]) - xk[1,0]*np.cos(xk[2,0])], 
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        
        scanA = np.vstack((mapped_scan_pose, [1]))
        scanA[2,0] = 0

        theta = -xk[2,0] + mapped_scan_pose[2,0]

        # Left-multiply the augmented array by the homogeneous matrix
        transformed_array = np.dot(invertedTransform, scanA)

        # Select all rows and all but the last column
        transformed_array = transformed_array[:-1, :]

        result = np.array([[np.cos(theta), -np.sin(theta), 0, transformed_array[0,0]], 
                       [np.sin(theta), np.cos(theta), 0, transformed_array[1,0]], 
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        return result
    
    

    def add_new_pose(self):
        """
        This function adds a new pose to the state vector and the covariance matrix

        :param xk_slam: the current state vector
        :type xk_slam: numpy array of shape (3n,1)
        :param Pk_slam: the current covariance matrix
        :type Pk_slam: numpy array of shape (3n,3n)
        :return: None
        """

        #clone the last three elements of xk_slam and add them to the pose history, use vstack to stack the arrays vertically
        self.xk_slam = np.vstack((self.xk_slam, self.xk_slam[-3:,:]))

        #publish the cloned pose
        # self.publish_cloned_pose_path(self.xk_slam[-3:,:])
        self.publish_cloned_pose_marker(self.xk_slam[-3:,:])

        # clone the last three columns of Pk_slam and add it to Pk_slam
        self.Pk_slam  = np.hstack((self.Pk_slam ,self.Pk_slam [:,-3:]))

        # clone the last three rows of Pk_slam and add them to Pk_slam
        self.Pk_slam = np.vstack((self.Pk_slam, self.Pk_slam[-3:,:]))
        
        pass
        


    def ICP_registration(self, new_scan_pose, mapped_scan_pose, new_scan, mapped_scan):
        """ this function computes the runs the ICP algorithm and calculate
            displacement  between the current pose and the pose at which the scan was taken

        :param new_scan_pose: the pose at which the scan was taken
        :type new_scan_pose: numpy array of shape (3,1) 
        :param mapped_scan_pose: the pose of the overlapping scan in the map
        :type mapped_scan_pose: numpy array of shape (3,1)
        :param new_scan_pcd: the scan to be registered
        :type new_scan_pcd: open3d.geometry.PointCloud
        :param mapped_scan_pcd: the overlapping scan in the map
        :type mapped_scan_pcd: open3d.geometry.PointCloud

        :return: the displacement between the current pose and the pose at which the scan was taken
        :rtype: numpy array of shape (3,1)
        """

        #convert the scans to point clouds
        new_scan_pcd = o3d.geometry.PointCloud()
        new_scan_pcd.points = o3d.utility.Vector3dVector(new_scan)

        mapped_scan_pcd = o3d.geometry.PointCloud()
        mapped_scan_pcd.points = o3d.utility.Vector3dVector(mapped_scan)

        # Compute initial displacement using the guess function
        #NOTE: previous version of icp used the guess function to compute the initial displacement 
        # but the final version does not use it
        trans_init = self.displacement_guess(new_scan_pose, mapped_scan_pose)

        # convert the transformation to float
        trans_init = trans_init.astype('float64')

        if self.ICP_method == 1:
            reg_p2p = o3d.pipelines.registration.registration_icp(new_scan_pcd, mapped_scan_pcd, max_correspondence_distance=self.registration_threshold)
        
        elif self.ICP_method == 2:
            reg_p2p = o3d.pipelines.registration.registration_icp(
            new_scan_pcd,mapped_scan_pcd, self.registration_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))

        elif self.ICP_method == 3:
            # Compute normal vectors for target point cloud
            mapped_scan_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # Compute normal vectors for source point cloud
            new_scan_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # Run global registration with normal vectors
            reg_p2p = o3d.pipelines.registration.registration_icp(
                new_scan_pcd, mapped_scan_pcd, self.registration_threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))

        else:
            print('usage: \n 1: point_to_point without initial guess \n 2: point_to_point with initial guess \n 3: point_to_plane with initial guess')   
            raise ValueError('Invalid ICP method chosen')
        
        # apply the obtained transformation to the new scan
        new_scan_pcd.transform(reg_p2p.transformation)

        # extract the heading angle from the transformation matrix
        theta = np.arctan2(reg_p2p.transformation[1,0], reg_p2p.transformation[0,0])
        scan_displacement = np.array([[reg_p2p.transformation[0,-1]], [reg_p2p.transformation[1,-1]], [theta]])

        Rk = self.ICP_Rk

        return scan_displacement, Rk 
    


    def jacobianFx(self, linear_velocity, theta, delta_time):
        """this function calculates the jacobian matrix of the robot's motion model with respect to the state vector

        :param linear_velocity: the linear velocity of the robot
        :type linear_velocity: float
        :param theta: the orientation of the robot
        :type theta: float
        :param delta_time: the time difference between the current and previous encoder readings
        :type delta_time: float

        :return: jacobianFx: the jacobian matrix of the robot's motion model with respect to the state vector
        :rtype: numpy array
        """

        Fx = np.array([[1, 0, -linear_velocity * np.sin(theta) * delta_time],     
                    [0, 1, linear_velocity * np.cos(theta) * delta_time],
                    [0, 0, 1]])
        return Fx
    


    def jacobianFw(self, theta, delta_time):
        """this function calculates the jacobian matrix of the robot's motion model with respect to the noise vector
        
        Args:
            theta (float): the orientation of the robot
            delta_time (float): the time difference between the current and previous encoder readings
            
        Returns:
            jacobainFw (numpy array): the jacobian matrix of the robot's motion model with respect to the noise vector
            
        """

        Fw = np.array([[self.wheel_radius* np.cos(theta)*delta_time/2, self.wheel_radius* np.cos(theta)*delta_time/2], 
                    [self.wheel_radius* np.sin(theta)*delta_time/2, self.wheel_radius* np.sin(theta)*delta_time/2], 
                    [self.wheel_radius * delta_time/self.wheel_distance, -self.wheel_radius * delta_time/self.wheel_distance]])
        
        return Fw
    
    


    def jacobianF1k(self, xk_slam, jacobianFx):
        """
        this function calculates the jacobian matrix of the robot's motion model with respect to the state vector
        
        :param xk_slam: the robot's pose history
        :type xk_slam: 2D numpy array
        :param jacobianFx: the jacobian matrix of the robot's motion model with respect to the state vector
        :type jacobianFx: 2D numpy array
        
        :return: the jacobian matrix of the robot's motion model with respect to the state vector
        :rtype: 2D numpy array
        """

        # count the number of rows in the xk_slam array and divide by 3
        # this will give the number of poses in the pose history
        num_poses = int(xk_slam.shape[0]/3)

        # create an identity matrix of size 3*num_poses x 3*num_poses#
        jacobianF1k = np.identity(3*num_poses)

        # replace the last 3 rows of the last 3 columns of the identity matrix with the jacobianFx
        jacobianF1k[-3:, -3:] = jacobianFx

        return jacobianF1k
    


    def jacobianF2k(self,xk_slam, jacobianFw):
        """ this function calculates the jacobian matrix of the robot's motion model with respect to the noise vector

        :param xk_slam: the robot's pose history
        :type xk_slam: 2D numpy array
        :param jacobianFw: the jacobian matrix of the robot's motion model with respect to the noise vector
        :type jacobianFw: 2D numpy array

        :return: the jacobian matrix of the robot's motion model with respect to the noise vector
        :rtype: 2D numpy array
        """

        # count the number of rows in the xk_slam array and divide by 3
        # this will give the number of poses in the pose history
        num_poses = int(xk_slam.shape[0]/3)

        # create a zero matrix of size 3*num_poses x 2
        jacobianF2k = np.zeros((3*num_poses, 2))

        # replace the last 3 rows of the last 2 columns of the zero matrix with the jacobianFw
        jacobianF2k[-3:, -2:] = jacobianFw

        return jacobianF2k
    
        
    def J2_oplus(self, xk_robot):
        """
        this function calculates the jacobian matrix of the robot's motion model with respect to the noise vector
        
        :param xk_robot: the robot's pose history
        :type xk_robot: 2D numpy array
        
        :return: the jacobian matrix of the robot's motion model with respect to the noise vector
        :rtype: 2D numpy array
        """

        scan_pose_jacobian = np.array([[np.cos(xk_robot[2,0]), np.sin(xk_robot[2,0]), 0],
                             [-np.sin(xk_robot[2,0]), np.cos(xk_robot[2,0]), 0],
                             [0, 0, 1]])
        return scan_pose_jacobian 
    


    def J_minus_J1_oplus(self, scan_index, xk_robot, xk_slam):
        """
        this function calculates the jacobian matrix of the robot's motion model with respect to the noise vector

        :param scan_index: the index of the scan pose in the xk_slam array
        :type scan_index: int
        :param xk_robot: the robot's pose history
        :type xk_robot: 2D numpy array
        :param xk_slam: the robot's pose history
        :type xk_slam: 2D numpy array

        :return: the jacobian matrix of the robot's motion model with respect to the noise vector
        :rtype: 2D numpy array
        """

        #extract the x,y and theta of the scan pose from the xk_slam array
        xk_scan = xk_slam[3*scan_index:3*scan_index+3, :]

        # 
        robot_pose_jacobian = np.array([[-np.cos(xk_robot[2,0]), -np.sin(xk_robot[2,0]), -np.sin(xk_robot[2,0])*(xk_scan[0,0]-xk_robot[0,0]) + np.cos(xk_robot[2,0])*(xk_scan[1,0]-xk_robot[1,0])],
                                [np.sin(xk_robot[2,0]), -np.cos(xk_robot[2,0]), -np.cos(xk_robot[2,0])*(xk_scan[0,0]-xk_robot[0,0])-np.sin(xk_robot[2,0])*(xk_scan[1,0]-xk_robot[1,0])],
                                [0, 0, -1]])
        
        return robot_pose_jacobian
    


    def jacobianHk(self,scan_index, xk_slam):
        """this function calculates the jacobian matrix of the robot's measurement model with respect to the state vector

        :param scan_index: the index of the scan pose in the xk_slam array
        :type scan_index: int
        :param xk_slam: the robot's pose history
        :type xk_slam: 2D numpy array

        :return: the jacobian matrix of the robot's measurement model with respect to the state vector
        :rtype: 2D numpy array
        """

        num_poses = int(xk_slam.shape[0]/3)

        #extract the last 3 rows of the xk_slam array and store them in a variable called xk
        xk_robot = xk_slam[-3:, :]

        #create a zero matrix of size 3 x 3*num_poses
        jacobianHk = np.zeros((3, 3*num_poses))

        #replace the last 3 rows of the last 3 columns of the zero matrix with the dh_dxk
        J_minus_J1_oplus = self.J_minus_J1_oplus(scan_index, xk_robot, xk_slam)

        # the last 6 columns of the jacobianHk matrix will be replaced with the J_minus_J1_oplus 
        # because the last 6 columns of the xk_slam array are the robot poses
        # and we are not predicting whenever update is called
        jacobianHk[:, -6:-3] = J_minus_J1_oplus

        jacobianHk[:, -3:] =  J_minus_J1_oplus

        #replace the all the rows of the column from 3*scan_index to 3*scan_index+3 with the j2_plus
        jacobianHk[:, 3*scan_index:3*scan_index+3] = self.J2_oplus(xk_robot)

        return jacobianHk

 

    def observation_eqn(self,xk_scan, xk_robot):
        """this function calculates the observation equation
        
        :param xk_scan: the scan pose
        :type xk_scan: 2D numpy array
        :param xk_robot: the robot pose
        :type xk_robot: 2D numpy array
        
        :return: the observation equation
        :rtype: 2D numpy array
        """
        
        hx = np.array([[(xk_scan[0,0] - xk_robot[0,0])*np.cos(xk_robot[2,0]) + (xk_scan[1,0] - xk_robot[1,0])*np.sin(xk_robot[2,0])],
                    [-(xk_scan[0,0] - xk_robot[0,0])*np.sin(xk_robot[2,0]) + (xk_scan[1,0] - xk_robot[1,0])*np.cos(xk_robot[2,0])],
                    [-xk_robot[2,0] + xk_scan[2,0]]])
        return hx

    

    def lidar_update(self, scan_index, measurement, Rk):
        """this function updates the state vector and the covariance matrix using the scan displacement measurement

        :param scan_index: the index of the scan in the state vector
        :type scan_index: int
        :param measurement: the scan displacement measurement
        :type measurement: 2D numpy array
        :param Rk: the covariance matrix of the measurement
        :type Rk: 2D numpy array

        :return: the updated state vector and the covariance matrix
        :rtype: 2D numpy array, 2D numpy array
        """

        num_poses = int(self.xk_slam.shape[0]/3)

        # calculate the predicted measurement and the jacobian of the measurement
        predicted_measurement = self.observation_eqn(self.xk_slam[3*scan_index:3*scan_index+3,:], self.xk_slam[-3:,:])

        # compute the jacobian of the observation
        Hk = self.jacobianHk(scan_index, self.xk_slam)

        #compute the innovation
        innovation = measurement - predicted_measurement

        #compute the innovation covariance
        S = (Hk @ self.Pk_slam @ Hk.T) + (self.ICP_Vk @ Rk @ self.ICP_Vk.T)

        # compute the mahalanobis distance D
        D = innovation.T @ np.linalg.inv(S) @ innovation

        #check if the mahalanobis distance is within the threshold
        if np.sqrt(D) <= self.chi_square_min_dist_thresh:

            #compute the kalman gain
            K = self.Pk_slam @ Hk.T @ np.linalg.inv(S)

            #log the predicted measurement using rospy logging
            print('predicted measurement: %s', predicted_measurement)

            #log the measurement using rospy logging
            print('measurement: %s', measurement)

            #log the innovation using rospy logging
            print('innovation: %s', innovation)


            # wrap the innovation angle between -pi and pi
            # innovation[1] = self.wrap_angle(innovation[2])

            #update the state vector
            self.xk_slam = self.xk_slam + np.dot(K, innovation)

            #create an identity matrix
            I = np.eye(3*num_poses)

            #update the covariance matrix
            self.Pk_slam = (I - K @ Hk) @ self.Pk_slam @ (I - K @ Hk).T


            #extract the robot's pose from the state vector
            self.xk = self.xk_slam[-3:, :]
            #UPDATE ONLY THE first and second elements of self.xk
            # self.xk[0,0] = self.xk_slam[-3,0]
            # self.xk[1,0] = self.xk_slam[-2,0]

            #extract the robot's covariance from the covariance matrix
            self.Pk = self.Pk_slam[-3:, -3:]
            # self.Pk[0:2,0:2] = self.Pk_slam[-3:-1, -3:-1]

            #publish the robot's belief
            self.publish_belief(self.xk, self.Pk)

        else:
            #delete the scan from the map and the state vector
            self.map.pop(-1)

            #delete the scan from the state vector
            self.xk_slam = np.delete(self.xk_slam, [-6,-5,-4], axis=0)

            #delete the scan from the covariance matrix
            self.Pk_slam = np.delete(self.Pk_slam, [-6,-5,-4], axis=0)
            self.Pk_slam = np.delete(self.Pk_slam, [-6,-5,-4], axis=1)
        pass





    def wrap_angle(self, angle):
        """this function wraps the angle between -pi and pi

        :param angle: the angle to be wrapped
        :type angle: float

        :return: the wrapped angle
        :rtype: float
        """
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )


    def do_scans_overlap(self, pcd1, pcd2):
        """
        This function checks if two point clouds expressed in the same reference frame overlaps

        :param pcd1: source point cloud
        :type pcd1: 2D numpy array
        :param pcd2: target point cloud
        :type  pcd2: 2D numpy array

        :o1: orginal pcd1
        :o2: original pcd2

        :return 
        """

        if self.overlap_method == 1:
            
            nbrs = NearestNeighbors(n_neighbors = 1, algorithm = "auto").fit(pcd1)
            distances, indices = nbrs.kneighbors(pcd2)

            if np.any(distances <= self.knn_min_dist_thresh):
                a = np.where(distances <= self.knn_min_dist_thresh)
        
                return True
            else:
                return False

        elif self.overlap_method == 2:
            scan1 = pcd1[:, :-1]

            scan2 = pcd2[:, :-1]

            # convert the scan to list
            scan1_coords_list = [(int(round(scan[0])), int(round(scan[1]))) for scan in scan1.tolist()]

            scan2_coords_list = [(int(round(scan[0])), int(round(scan[1]))) for scan in scan2.tolist()]

            #convert the scan to polygon 
            scan1_polygon = Polygon(scan1_coords_list)

            #convert the scan to polygon
            scan2_polygon = Polygon(scan2_coords_list)


            if not scan1_polygon.is_valid:
                scan1_polygon = scan1_polygon.buffer(0)

            if not scan2_polygon.is_valid:
                scan2_polygon = scan2_polygon.buffer(0)

            #check if the two scans overlap
            overlap_area = scan1_polygon.intersection(scan2_polygon).area

            smaller_area = min(scan1_polygon.area, scan2_polygon.area)

            #compute the overlap percentage
            overlap_percentage = overlap_area / smaller_area * 100

            #check if the overlap percentage is greater than the threshold
            if overlap_percentage >= self.overlap_threshold:
                return True
            
            return False
        
        else:
            print("usage: \n 1: knn method \n 2: polygon method")
            raise ValueError("Invalid scan overlap method chosen")




    def publish_robot_traj(self, xk) -> None:
        """this function publishes the robot's path as a path message

        :param xk: the robot's pose
        :type xk: 2D numpy array

        :return: None
        :rtype: None
        """

        #publish the gps measurement as a path
        self.robot_path_msg.header.frame_id = "world_ned" 
        self.robot_path_msg.header.stamp = rospy.Time.now()
        pose = PoseStamped()
        pose.header.frame_id = "world_ned"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = xk[0,0]
        pose.pose.position.y = xk[1,0]
        pose.pose.position.z = 0.0


        #convert the yaw angle to a euler to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, xk[2,0])
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]


        #to visulaize the path in plotjuggler, uncomment lines 1-2, 4 and comment line 3
        self.robot_path_msg.poses.append(pose)        

        #publish the robot's path
        self.pub_robot_path.publish(self.robot_path_msg)
        pass


    def publish_cloned_pose_path(self, xk_cloned) -> None:
        """this function publishes the robot's path as a path message

        :param xk: the robot's pose
        :type xk: 2D numpy array

        :return: None
        :rtype: None
        """
        #publish the gps measurement as a path
        self.cloned_pose_path_msg.header.frame_id = "world_ned" 
        self.cloned_pose_path_msg.header.stamp = rospy.Time.now()
        pose = PoseStamped()
        pose.header.frame_id = "world_ned"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = xk_cloned[0,0]
        pose.pose.position.y = xk_cloned[1,0]
        pose.pose.position.z = 0.0


        #convert the yaw angle to a euler to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, xk_cloned[2,0])
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]


        #to visulaize the path in plotjuggler, uncomment lines 1-2, 4 and comment line 3
        self.cloned_pose_path_msg.poses.append(pose)        

        #publish the robot's path
        # self.pub_cloned_pose_path.publish(self.cloned_pose_path_msg)
        pass


    def cov_publish_belief(self, xk, Pk) -> None:
            """this function publishes the robot's pose as an odometry message
            """
            # create an odometry message
            n = int(xk.shape[0] / 3) # number of landmarks
            for i in range(n):

                if i == (n-1):
                    continue
                self.Cov.header.stamp = rospy.Time.now()
                self.Cov.header.frame_id = "world_ned"
                self.Cov.child_frame_id = "kobuki/base_footprint"
                self.Cov.pose.pose.position.x = xk[3*i,0]    
                self.Cov.pose.pose.position.y = xk[3*i + 1,0]
                self.Cov.pose.pose.position.z = 0#xk[3*i + 2,0]

                # #convert the yaw angle to a quaternion
                orientation = quaternion_from_euler(0, 0, xk[3*i + 2,0])
                self.kobuki_pose.pose.pose.orientation.x = orientation[0]
                self.kobuki_pose.pose.pose.orientation.y = orientation[1]
                self.kobuki_pose.pose.pose.orientation.z = orientation[2]
                self.kobuki_pose.pose.pose.orientation.w = orientation[3]

                #initialize the covariance matrix
                self.Cov.pose.covariance = [Pk[3*i,3*i], Pk[3*i,3*i + 1], 0, 0, 0, Pk[3*i,3*i + 2], 
                                                    Pk[3*i + 1, 3*i], Pk[3*i + 1, 3*i + 1,], 0, 0, 0, Pk[3*i + 1, 3*i + 2], 
                                                    0, 0, 0, 0, 0, 0, 
                                                    0, 0, 0, 0, 0, 0, 
                                                    0, 0, 0, 0, 0, 0, 
                                                    Pk[3*i + 2, 3*i], Pk[3*i + 2, 3*i + 1], 0, 0, 0, Pk[3*i + 2, 3*i + 2]]

                #publish the robot's pose as odometry messages
                self.pub1.publish(self.Cov)
            pass 


    def publish_cloned_pose_marker(self, xk_cloned):
        """this function publishes the pose at which the scan was taken as a marker

        :param xk_cloned: the pose at which the scan was taken
        :type xk_cloned: 2D numpy array

        :return: None
        :rtype: None
        """
        
        marker = Marker()
        #publish the pose as a marker

        marker.header.frame_id = "world_ned"
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(214748364)
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.ns = "cloned_pose"
        marker.id = self.cloned_marker_id
        marker.pose.position.x = xk_cloned[0,0]
        marker.pose.position.y = xk_cloned[1,0]
        marker.pose.position.z = 0.0

        marker.scale.x = 0.04
        marker.scale.y = 0.04
        marker.scale.z = 0.04
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker.color.a = 1.0

        
        #convert the yaw angle to a euler to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, xk_cloned[2,0])
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        
        self.pub_cloned_pose_marker.publish(marker)
        self.cloned_marker_id += 1
        pass



    def ground_truth_odom_callback(self, msg) -> None:
        #publish the ground truth odometry as a path
        self.ground_truth_odom_path_msg.header.frame_id = "world_ned"
        self.ground_truth_odom_path_msg.header.stamp = rospy.Time.now()
        pose = PoseStamped()
        pose.header.frame_id = "world_ned"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = msg.pose.pose.position.x
        pose.pose.position.y = msg.pose.pose.position.y
        pose.pose.position.z = 0.0

        #convert the yaw angle to a euler to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, msg.pose.pose.orientation.z)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        
        self.ground_truth_odom_path_msg.poses.append(pose)

        #publish the ground truth odometry path
        self.pub_ground_truth_odom_path.publish(self.ground_truth_odom_path_msg)

        gt = np.array([[msg.pose.pose.position.x, msg.pose.pose.position.y]])

        pass



    def transform_pcd(self, pcd, xk):
        """
        this function applies a transformation on a numpy array of point clouds and returns the 
        transformed point cloud in the new frame

        :param pcd: the cloud of points to be transformed
        :type pcd: 2D numpy array
        :param xk: rthe robot's current pose
        :type xk: 2D numpy array
        """

        augmented_array = np.ones((pcd.shape[0], 4))
        augmented_array[:,:-1] = pcd

        # compute the transformation matrix btw between the robot and the lidar
        wTr = np.array([[np.cos(xk[2,0]), -np.sin(xk[2,0]), 0, xk[0,0] ], 
                [np.sin(xk[2,0]), np.cos(xk[2,0]), 0, xk[1,0]] , 
                [0, 0, 1, 0], 
                [0, 0, 0, 1]])
        
        robotTlidar = np.array([[np.cos(self.robotTlidar[3,0]), -np.sin(self.robotTlidar[3,0]), 0.0, self.robotTlidar[0,0]],
                                [np.sin(self.robotTlidar[3,0]), np.cos(self.robotTlidar[3,0]), 0.0, self.robotTlidar[1,0]],
                                [0.0, 0.0, 1.0, self.robotTlidar[2,0]],
                                [0.0, 0.0, 0.0, 1.0]])
        
        rotation = np.dot(wTr, robotTlidar)

        # Left-multiply the augmented array by the homogeneous matrix
        transformed_array = np.dot(rotation, augmented_array.T).T

        # Select all rows and all but the last column
        transformed_array = transformed_array[:, :-1]

        return transformed_array
    
    


    def publish_point_cloud(self, point_cloud1, point_cloud2):
        """Publishes a point cloud message to the topic /point_cloud
        
        :param point_cloud1: the point cloud to be published
        :type point_cloud1: 2D numpy array
        :param point_cloud2: the point cloud to be published
        :type point_cloud2: 2D numpy array
        """
        
            
        # Create header for point cloud message
        header1 = Header()
        header1.frame_id = "world_ned"
        header1.stamp = rospy.Time.now()

        header2 = Header()
        header2.frame_id = "world_ned"
        header2.stamp = rospy.Time.now()

        # Create fields for point cloud message
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        # Create point cloud message
        cloud_msg1 = ab.create_cloud(header1, fields, point_cloud1)
        cloud_msg2 = ab.create_cloud(header2, fields, point_cloud2)
        self.point_cloud1_pub.publish(cloud_msg1)
        self.point_cloud2_pub.publish(cloud_msg2)

        pass

    def publish_all_aligned_scans(self, point_cloud1):
            
            # Create header for point cloud message
            header1 = Header()
            header1.frame_id = "world_ned"
            header1.stamp = rospy.Time.now()


            # Create fields for point cloud message
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)
            ]

            # Create point cloud message
            cloud_msg1 = ab.create_cloud(header1, fields, point_cloud1)
            self.all_aligned_scans_pub.publish(cloud_msg1)

            pass



    def publish_belief(self, xk, Pk) -> None:
        """this function publishes the robot's pose as an odometry message

        :param xk: the robot's current pose
        :type xk: 2D numpy array
        :param Pk: the robot's current covariance matrix
        :type Pk: 2D numpy array
        """
        # create an odometry message
        self.kobuki_pose.header.stamp = rospy.Time.now()
        self.kobuki_pose.header.frame_id = "world_ned"
        self.kobuki_pose.child_frame_id = "kobuki/base_footprint"
        self.kobuki_pose.pose.pose.position.x = xk[0,0]
        self.kobuki_pose.pose.pose.position.y = xk[1,0]
        self.kobuki_pose.pose.pose.position.z = 0

        #convert the yaw angle to a quaternion
        orientation = quaternion_from_euler(0, 0, xk[2,0])
        self.kobuki_pose.pose.pose.orientation.x = orientation[0]
        self.kobuki_pose.pose.pose.orientation.y = orientation[1]
        self.kobuki_pose.pose.pose.orientation.z = orientation[2]
        self.kobuki_pose.pose.pose.orientation.w = orientation[3]

        #set the linear and angular velocities
        self.kobuki_pose.twist.twist.linear.x = self.linear_velocity
        self.kobuki_pose.twist.twist.linear.y = 0
        self.kobuki_pose.twist.twist.linear.z = 0
        self.kobuki_pose.twist.twist.angular.x = 0
        self.kobuki_pose.twist.twist.angular.y = 0
        self.kobuki_pose.twist.twist.angular.z = self.angular_velocity


        #initialize the covariance matrix
        self.kobuki_pose.pose.covariance = [Pk[0,0], Pk[0,1], 0, 0, 0, Pk[0,2], 
                                            Pk[1,0], Pk[1,1], 0, 0, 0, Pk[1,2], 
                                            0, 0, 0, 0, 0, 0, 
                                            0, 0, 0, 0, 0, 0, 
                                            0, 0, 0, 0, 0, 0, 
                                            Pk[2,0], Pk[2,1], 0, 0, 0, Pk[2,2]]

        #publish the robot's pose as odometry messages
        self.pub.publish(self.kobuki_pose)
        pass 



    def start(self):
        """this function is called to start the localization process
        """
        rospy.loginfo("Localization started")
        rospy.spin()
        pass



if __name__ == '__main__':
    try:
        print("Starting the node")

        # Initialize ROS node
        rospy.init_node("kobuki_controller") 

        print("creating the kobuki controller")
        odom_ = PoseBasedSlam()

        odom_.start()


    except rospy.ROSInterruptException:
        pass


