
WIDTH = 640
HEIGHT = 480
FPS = 60

HOME_JOINTS = [-0.06173697, -0.82587762, 0.04173763, -2.45493389, 0.0306236,  1.62901584, 0.75014108]
HOME_EE_T = [0.30, 0.00, 0.45]
HOME_EE_Q = [1.00 , 0.00 , 0.00 , 0.00]
HOME_EE_POSE = HOME_EE_T + HOME_EE_Q

START_JOINTS = [ 0.10541233, -0.62125134, -0.17258245, -1.85871562, -0.15030898 , 1.75584733, 0.77609249]
TASK_DICT = {
    "test": "Do something",
    "pick_block": "Pick up the red block on the table.",
}

START_DICT = {
    "pick_block": [ 0.16827892,  0.3525368 , -0.04366834, -1.70639706, -0.00772997, 2.11010814,  0.93295145],
}


ROBOT_SPEED_FACTOR = 0.1
GRIPPER_SPEED = 0.03 # [m/s] 
GRIPPER_FORCE = 15.0 # [N] 
# self.robot.translation_velocity_limit.set(3.0)
# self.robot.rotation_velocity_limit.set(2.5)
# self.robot.elbow_velocity_limit.set(2.62)
# self.robot.translation_acceleration_limit.set(9.0)
# self.robot.rotation_acceleration_limit.set(17.0)
# self.robot.elbow_acceleration_limit.set(10.0)
# self.robot.translation_jerk_limit.set(4500.0)
# self.robot.rotation_jerk_limit.set(8500.0)
# self.robot.elbow_jerk_limit.set(5000.0)
# self.robot.joint_velocity_limit.set([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
# self.robot.joint_acceleration_limit.set([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
# self.robot.joint_jerk_limit.set([5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0])


# DT = 0.001