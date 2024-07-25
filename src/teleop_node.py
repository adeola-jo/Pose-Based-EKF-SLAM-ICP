#! /usr/bin/env python3


import rospy
from std_msgs.msg import Float64MultiArray

class TeleopRobot:
    def __init__(self):
        rospy.init_node('teleop_robot')

        self.publisher = rospy.Publisher('/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=1)
        # self.publisher = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=1)

        self.vel_msg = Float64MultiArray()
        self.left_velocity = 0.0
        self.right_velocity = 0.0

    def increase_left_velocity(self):
        self.left_velocity += 0.2

    def decrease_left_velocity(self):
        self.left_velocity -= 0.2


    def increase_right_velocity(self):
        self.right_velocity += 0.2
        print(f'velocity: [{self.left_velocity}, {self.right_velocity}]')


    def decrease_right_velocity(self):
        self.right_velocity -= 0.2


    def increase_both_velocities(self):
        self.left_velocity += 0.2
        self.right_velocity += 0.2


    def decrease_both_velocities(self):
        self.left_velocity -= 0.2
        self.right_velocity -= 0.2

    def stop_robot(self):
        self.left_velocity = 0.0
        self.right_velocity = 0.0


    def publish_velocities(self):
        self.vel_msg.data = [self.left_velocity, self.right_velocity]
        self.publisher.publish(self.vel_msg)
        print(f'velocity: [{self.left_velocity}, {self.right_velocity}]')
        print('------------------------------')


    def run(self):
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            key = input("Enter a key: ")

            if key == 'a':
                self.increase_left_velocity()
            elif key == 'b':
                self.decrease_left_velocity()
            elif key == 'c':
                self.increase_right_velocity()
            elif key == 'd':
                self.decrease_right_velocity()
            elif key == 'e':
                self.increase_both_velocities()
            elif key == 'f':
                self.decrease_both_velocities()
            elif key == 's':
                self.stop_robot()
            elif key == 'q':
                break
            else:
                continue

            self.publish_velocities()
            rate.sleep()

if __name__ == '__main__':
    
    teleop = TeleopRobot()

    print('\n')  
          
    print('''Welcome to our Hands-On Localization Presentation!
    ------------------------------------------------------
    Title: Pose-Based SLAM using ICP and EKF

    Instructions:
    - Press 'a' to increase just the left wheel velocities.
    - Press 'b' to decrease just the left wheel velocities.
    - Press 'c' to increase just the right wheel velocity.
    - Press 'd' to decrease just the right wheel velocity.
    - Press 'e' to increase both wheel velocities equally.
    - Press 'f' to decrease both wheel velocities equally.
    - Press 's' to stop the robot.
    - Press 'q' to quit the program.

    linear_velocity: [left, right]
    Please enter a key to control the robot:''')

    teleop.run()
