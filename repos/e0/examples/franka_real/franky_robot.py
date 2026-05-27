import math
import os
import time
from functools import wraps

import franky
import numpy as np
from franky import *
# from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

from utils import *



def retry_on_exception(wait_seconds=3, max_retries=5):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(self, *args, **kwargs)
                except franky._franky.ControlException as e:
                    if "Reflex" in str(e):
                        print(f"[FATAL] Robot entered Reflex mode: {e}")
                        print("[ACTION] try recover_from_errors() ...")
                        try:
                            ok = self.robot.recover_from_errors()
                            if ok:
                                print("[INFO] successed")
                            else:
                                print("[WARN] failed, reset Robot/Gripper")
                                self.robot = franky.Robot(self.robot_ip)
                                self.gripper = franky.Gripper(self.robot_ip)
                        except Exception as reset_e:
                            self.robot = franky.Robot(self.robot_ip)
                            self.gripper = franky.Gripper(self.robot_ip)

                        retries += 1
                        if max_retries is not None and retries >= max_retries:
                            raise
                        time.sleep(wait_seconds)
                    else:
                        retries += 1
                        print(f"[ERROR] {func.__name__}: {e}")
                        if max_retries is not None and retries >= max_retries:
                            raise
                        time.sleep(wait_seconds)
        return wrapper
    return decorator




class FrankaFrankyRobot():
    # def __init__(self, robot_ip = "172.16.0.2"):
    def __init__(self, robot_ip = "your robot ip"):
        self.robot_ip = robot_ip
        self.robot = franky.Robot(self.robot_ip)
        self.gripper = franky.Gripper(self.robot_ip)
        self.init_settings() 


    def init_settings(self,):
        self.robot_speed_factor = ROBOT_SPEED_FACTOR
        self.gripper_speed = GRIPPER_SPEED # [m/s] 
        self.gripper_force = GRIPPER_FORCE # [N] 
        self.home_joints_list = HOME_JOINTS
        self.home_pose_list = HOME_EE_POSE

       
        self.robot.relative_dynamics_factor = self.robot_speed_factor
        # self.robot.relative_dynamics_factor = RelativeDynamicsFactor(velocity=0.1, acceleration=0.05, jerk=0.1)

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

        # Get the max of each limit (as provided by Franka) with the max function, e.g.:
        # print(self.robot.joint_jerk_limit.max)


    def get_all_state(self,):
        state = self.robot.state
        # Get the robot's cartesian state
        cartesian_state = self.robot.current_cartesian_state
        robot_pose = cartesian_state.pose  # Contains end-effector pose and elbow position
        ee_pose = robot_pose.end_effector_pose
        elbow_pos = robot_pose.elbow_state
        robot_velocity = cartesian_state.velocity  # Contains end-effector twist and elbow velocity
        ee_twist = robot_velocity.end_effector_twist
        elbow_vel = robot_velocity.elbow_velocity
        
        # Get the robot's joint state
        joint_state = self.robot.current_joint_state
        joint_pos = joint_state.position
        joint_vel = joint_state.velocity


    @retry_on_exception()
    def home_joints(self,):
        print(f"Move to home joints: {self.home_joints_list}")
        self.set_joints(self.home_joints_list)


    @retry_on_exception()
    def home_pose(self,):
        print(f"Move to home pose: {self.home_pose_list}")
        self.set_ee_pose(self.home_pose_list)


    ##################################### motion #######################################
    ####################################################################################

    @retry_on_exception()
    def set_ee_pose(self, target_pose, async_mode = False):
        print(f"Move to target ee_pose: {target_pose}")
        motion = CartesianMotion(Affine(translation=target_pose[:3], quaternion=target_pose[-4:]))
        self.robot.move(motion, asynchronous = async_mode)
        # self.robot.join_motion() 


    @retry_on_exception()
    def set_ee_pose_relative(self, relative_xyz, async_mode = False):
        print(f"Move to relative ee_pose : x {relative_xyz[0]} | y {relative_xyz[1]} | z {relative_xyz[2]}")
        motion = CartesianMotion(Affine(relative_xyz), ReferenceType.Relative)
        self.robot.move(motion, asynchronous = async_mode)


    @retry_on_exception()
    def set_joints(self, target_joints, async_mode = False):
        print(f"Move to target joints: {target_joints}")
        motion = JointMotion(target=target_joints)
        self.robot.move(motion, asynchronous = async_mode)


    @retry_on_exception()
    def set_ee_pose_smooth(self, target_pose, steps=10, async_mode=False):

        current_xyz = np.array(self.robot.current_pose.end_effector_pose.translation)
        current_quat = np.array(self.robot.current_pose.end_effector_pose.quaternion)
        target_xyz  = np.array(target_pose[:3])
        target_quat = np.array(target_pose[-4:])

        print(f"[INFO] Current EE pose: {current_xyz}, {current_quat}")
        print(f"[INFO] Target  EE pose: {target_xyz}, {target_quat}")


        delta_xyz = target_xyz - current_xyz

        
        r_current = R.from_quat(current_quat)
        r_target  = R.from_quat(target_quat)

        waypoints = []
        for i in range(steps + 1):
            alpha = i / steps
            alpha_smooth = 0.5 - 0.5 * np.cos(np.pi * alpha)

            pos = (1 - alpha_smooth) * current_xyz + alpha_smooth * target_xyz
            rot = R.slerp(0, 1, [r_current, r_target])(alpha_smooth).as_quat()

            wp = CartesianWaypoint(Affine(translation=pos, quaternion=rot))
            waypoints.append(wp)

        motion = CartesianWaypointMotion(waypoints)

        print(f"[INFO] Executing smooth EE motion with {len(waypoints)} waypoints...")
        self.robot.move(motion, asynchronous=async_mode)


    @retry_on_exception()
    def set_ee_pose_relative_smooth(self, relative_xyz, steps=5, async_mode=False):

        current_xyz = np.array(self.robot.current_pose.end_effector_pose.translation)
        target_xyz  = current_xyz + np.array(relative_xyz)

        waypoints = []
        for i in range(steps + 1):
            alpha = i / steps
            alpha_smooth = 0.5 - 0.5 * np.cos(np.pi * alpha)

            pos = (1 - alpha_smooth) * current_xyz + alpha_smooth * target_xyz
            wp = CartesianWaypoint(Affine(translation=pos, quaternion=self.robot.current_pose.end_effector_pose.quaternion))
            waypoints.append(wp)

        motion = CartesianWaypointMotion(waypoints)

        print(f"[INFO] Executing smooth relative EE motion with {len(waypoints)} waypoints...")
        self.robot.move(motion, asynchronous=async_mode)



    @retry_on_exception()
    def set_joints_smooth(self, target_joints, steps=5, async_mode=False):

        current_qpos = np.array(self.robot.current_joint_positions)
        target_qpos = np.array(target_joints)

        print(f"[INFO] Current joints: {current_qpos}")
        print(f"[INFO] Target joints:  {target_qpos}")

        delta = target_qpos - current_qpos

        JOINT_VEL_LIMIT = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
        time_needed = np.max(np.abs(delta) / (JOINT_VEL_LIMIT * 0.5))
        steps = int(max(steps, time_needed * 50))  

        waypoints = []
        for i in range(steps + 1):
            alpha = i / steps
            alpha_smooth = 0.5 - 0.5 * np.cos(np.pi * alpha)
            pos = (1 - alpha_smooth) * current_qpos + alpha_smooth * target_qpos
            wp = JointWaypoint(pos.tolist())
            waypoints.append(wp)

        motion = JointWaypointMotion(waypoints)

        print(f"[INFO] Executing smooth motion with {len(waypoints)} waypoints...")
        self.robot.move(motion, asynchronous=async_mode)


    ############################### callback && reaction ###############################
    ####################################################################################


    def _callback_example(self,):

        def cb( robot_state: RobotState, time_step: Duration, rel_time: Duration, abs_time: Duration, control_signal: JointPositions,):
            print(f"At time {abs_time}, the target joint positions were {control_signal.q}")

        m_jp1 = JointMotion([-0.06173697 ,-0.82587762 , 0.04173763, -2.45493389 , 0.0306236 ,  1.62901584 , 0.75014108])
        m_jp1.register_callback(cb)
        self.robot.move(m_jp1)


    def _reaction_example(self,):

        motion = CartesianMotion(Affine([0.0, 0.0, 0.1]), ReferenceType.Relative)  # Move down 10cm
 
        # It is important that the reaction motion uses the same control mode as the original motion.
        # Hence, we cannot register a JointMotion as a reaction motion to a CartesianMotion.
        # Move up for 1cm
        reaction_motion = CartesianMotion(Affine([0.0, 0.0, -0.01]), ReferenceType.Relative)
        
        # Trigger reaction if the Z force is greater than 30N
        reaction = Reaction(Measure.FORCE_Z > 5.0, reaction_motion)
        motion.add_reaction(reaction)

        self.robot.move(motion)


        def reaction_callback(robot_state: RobotState, rel_time: float, abs_time: float):
            print(f"Reaction fired at {abs_time}.")
        reaction.register_callback(reaction_callback)



    ##################################### gripper #######################################
    #####################################################################################

    @retry_on_exception()
    def set_gripper(self, width):
        if width > 0.08:
            print(f"You cannot set gripper width to {width}, it will be set to 0.08 !")
            width = 0.08
        elif width < 0:
            print(f"You cannot set gripper width to {width}, it will be set to 0.00 !")
            width = 0.00

        success = self.gripper.move(width = width, speed = self.gripper_speed)
        if success:
            print(f"Now gripper width is {width}")


    @retry_on_exception()
    def set_gripper_async(self, width):
        success_future = self.gripper.move_async(width, self.gripper_speed)
        # Wait for 1s
        if success_future.wait(1):
            print(f"Success: {success_future.get()}")
        else:
            self.gripper.stop()
            success_future.wait()
            print("Gripper motion timed out.")


    @retry_on_exception()
    def grasp(self, width = 0.0, epsilon_outer=0.1):
        print("Try to grasp object")
        success = self.gripper.grasp(width, self.gripper_speed, self.gripper_force, epsilon_outer=epsilon_outer)
        if success :
            print(f"Successfully grasped the object, width is {self.gripper.width}")
        else:
            print("Failed to grasped the object")
            self.open()
        return success


    @retry_on_exception()
    def open(self,):
        self.gripper.open(self.gripper_speed)


    ###################################### other #######################################
    ####################################################################################
    def compute_kinematics(self, qpos):

        f_t_ee = Affine()
        ee_t_k = Affine()
        ee_pose = self.robot.model.pose(Frame.EndEffector, qpos, f_t_ee, ee_t_k)
        return ee_pose
    

    @property
    def gripper_width(self,):
        return self.gripper.width

    @property
    def ee_pose(self,):
        ee_t = self.robot.current_pose.end_effector_pose.translation
        ee_q = self.robot.current_pose.end_effector_pose.quaternion
        return np.append(ee_t, ee_q)
    
    @property
    def ee_t(self,):
        return self.robot.current_pose.end_effector_pose.translation
    
    @property
    def ee_q(self,):
        return self.robot.current_pose.end_effector_pose.quaternion

    @property
    def qpos(self,):
        return self.robot.current_joint_positions
    
    @property
    def qvel(self,):
        return self.robot.current_joint_velocities
    
    @property
    def jacobian(self,):
        state = self.robot.state
        jocabian =  self.robot.model.body_jacobian(Frame.EndEffector, state)
        return jocabian



"""  四种控制模式

======================================= 1.  Joint Position Control =======================================

# A point-to-point motion in the joint space
m_jp1 = JointMotion([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7])
 
# A motion in joint space with multiple waypoints. The robot will stop at each of these
# waypoints. If you want the robot to move continuously, you have to specify a target velocity
# at every waypoint as shown in the example following this one.
m_jp2 = JointWaypointMotion(
    [
        JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
        JointWaypoint([0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8]),
        JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9]),
    ]
)
 
# Intermediate waypoints also permit to specify target velocities. The default target velocity
# is 0, meaning that the robot will stop at every waypoint.
m_jp3 = JointWaypointMotion(
    [
        JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
        JointWaypoint(
            JointState(
                position=[0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8],
                velocity=[0.1, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
            )
        ),
        JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9]),
    ]
)
 
# Stop the robot in joint position control mode. The difference of JointStopMotion to other
# stop-motions such as CartesianStopMotion is that JointStopMotion stops the robot in joint
# position control mode while CartesianStopMotion stops it in cartesian pose control mode. The
# difference becomes relevant when asynchronous move commands are being sent or reactions are
# being used(see below).
m_jp4 = JointStopMotion()


======================================= 2.  Joint Velocity Control =======================================

# Accelerate to the given joint velocity and hold it. After 1000ms stop the robot again.
m_jv1 = JointVelocityMotion(
    [0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4], duration=Duration(1000)
)
 
# Joint velocity motions also support waypoints. Unlike in joint position control, a joint
# velocity waypoint is a target velocity to be reached. This particular example first
# accelerates the joints, holds the velocity for 1s, then reverses direction for 2s, reverses
# direction again for 1s, and finally stops. It is important not to forget to stop the robot
# at the end of such a sequence, as it will otherwise throw an error.
m_jv2 = JointVelocityWaypointMotion(
    [
        JointVelocityWaypoint(
            [0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4], hold_target_duration=Duration(1000)
        ),
        JointVelocityWaypoint(
            [-0.1, -0.3, 0.1, -0.0, -0.1, 0.2, -0.4],
            hold_target_duration=Duration(2000),
        ),
        JointVelocityWaypoint(
            [0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4], hold_target_duration=Duration(1000)
        ),
        JointVelocityWaypoint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
)
 
# Stop the robot in joint velocity control mode.
m_jv3 = JointVelocityStopMotion()


======================================= 3.  Cartesian Position Control =======================================

# Move to the given target pose
quat = Rotation.from_euler("xyz", [0, 0, math.pi / 2]).as_quat()
m_cp1 = CartesianMotion(Affine([0.4, -0.2, 0.3], quat))
 
# With target elbow angle (otherwise, the Franka firmware will choose by itself)
m_cp2 = CartesianMotion(
    RobotPose(Affine([0.4, -0.2, 0.3], quat), elbow_state=ElbowState(0.3))
)
 
# A linear motion in cartesian space relative to the initial position
# (Note that this motion is relative both in position and orientation. Hence, when the robot's
# end-effector is oriented differently, it will move in a different direction)
m_cp3 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
 
# Generalization of CartesianMotion that allows for multiple waypoints. The robot will stop at
# each of these waypoints. If you want the robot to move continuously, you have to specify a
# target velocity at every waypoint as shown in the example following this one.
m_cp4 = CartesianWaypointMotion(
    [
        CartesianWaypoint(
            RobotPose(Affine([0.4, -0.2, 0.3], quat), elbow_state=ElbowState(0.3))
        ),
        # The following waypoint is relative to the prior one and 50% slower
        CartesianWaypoint(
            Affine([0.2, 0.0, 0.0]),
            ReferenceType.Relative,
            RelativeDynamicsFactor(0.5, 1.0, 1.0),
        ),
    ]
)
 
# Cartesian waypoints permit to specify target velocities
m_cp5 = CartesianWaypointMotion(
    [
        CartesianWaypoint(Affine([0.5, -0.2, 0.3], quat)),
        CartesianWaypoint(
            CartesianState(
                pose=Affine([0.4, -0.1, 0.3], quat), velocity=Twist([-0.01, 0.01, 0.0])
            )
        ),
        CartesianWaypoint(Affine([0.3, 0.0, 0.3], quat)),
    ]
)
 
# Stop the robot in cartesian position control mode.
m_cp6 = CartesianStopMotion()


======================================= 4.  Cartesian Velocity Control =======================================

# A cartesian velocity motion with linear (first argument) and angular (second argument)
# components
m_cv1 = CartesianVelocityMotion(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]))
 
# With target elbow velocity
m_cv2 = CartesianVelocityMotion(
    RobotVelocity(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]), elbow_velocity=-0.2)
)
 
# Cartesian velocity motions also support multiple waypoints. Unlike in cartesian position
# control, a cartesian velocity waypoint is a target velocity to be reached. This particular
# example first accelerates the end-effector, holds the velocity for 1s, then reverses
# direction for 2s, reverses direction again for 1s, and finally stops. It is important not to
# forget to stop the robot at the end of such a sequence, as it will otherwise throw an error.
m_cv4 = CartesianVelocityWaypointMotion(
    [
        CartesianVelocityWaypoint(
            Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]),
            hold_target_duration=Duration(1000),
        ),
        CartesianVelocityWaypoint(
            Twist([-0.2, 0.1, -0.1], [-0.1, 0.1, -0.2]),
            hold_target_duration=Duration(2000),
        ),
        CartesianVelocityWaypoint(
            Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]),
            hold_target_duration=Duration(1000),
        ),
        CartesianVelocityWaypoint(Twist()),
    ]
)
 
# Stop the robot in cartesian velocity control mode.
m_cv6 = CartesianVelocityStopMotion()

"""



if __name__=="__main__":
    fr_robot = FrankaFrankyRobot()
    # fr_robot.home_joints()
    print(fr_robot.qpos)
    print(fr_robot.ee_pose)

