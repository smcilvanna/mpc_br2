#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import numpy as np
import time
# from matplotlib import pyplot as plt
# import matplotlib.animation as animation
# import matplotlib.patches as mpatches
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pymavlink import mavutil
from pymap3d import geodetic2ned
import rospy

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3

# >>>>>> FUNCTIONS <<<<<<<<

def sys_arm():
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
    )
    print("Waiting to arm...")
    master.motors_armed_wait()
    print("Armed")

def wraptopi(x):   # used to wrap angle errors to interval [-pi pi]
     
    x = x - np.floor(x/(2*np.pi)) *2*np.pi
    if x > np.pi:
        x = x - 2*np.pi
    return x

def map_range(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

def set_rc_channel_pwm(channel_id, pwm=1500):
    # here: https://www.ardusub.com/operators-manual/rc-input-and-output.html#rc-inputs
    if channel_id < 1 or channel_id > 18:
        print("Channel does not exist.")
        return

    # https://mavlink.io/en/messages/common.html#RC_CHANNELS_OVERRIDE
    rc_channel_values = [65535 for _ in range(18)]
    rc_channel_values[channel_id - 1] = pwm
    master.mav.rc_channels_override_send(
        master.target_system,                # target_system
        master.target_component,             # target_component
        *rc_channel_values)                  # RC channel list, in microseconds.

def local_pos(lat, lon):
    global lat0, lng0
    north, east, down = geodetic2ned(lat=lat, lon=lon, lat0=lat0, lon0=lng0, h=0, h0=0)
    return north, east

def listenCB(msg):
    global dlat, dlon, flag_subscriber
    dlat = msg.latitude
    dlon = msg.longitude
    flag_subscriber = True

def getFormation(msg):
    global osetE, osetN, rov_id
    if msg.z == rov_id:
        osetN = msg.x
        osetE = msg.y

def tau2pwm(tau):
    # 'AUTO_Linear': {'translational': (358.84, 0.2428), 'rotational': (56.697, 0.0384)},
    # 'MAN_Linear': {'translational': (416.25, 0.2775), 'rotational': (66.15, 0.0441)}
    pwm = np.zeros((6,1))
    for idx, ctrl in enumerate(tau):
        if idx < 4: # translational
            pwm[idx] = (ctrl + 416.25) / 0.2775
        else: # rotational
            pwm[idx] = (ctrl + 66.15) / 0.0441

        dz = 25

        if pwm[idx] > (1500-dz) and pwm[idx] < (1500+dz):
            pwm[idx] = 1500
        
    pwm = pwm.astype(int)
    return pwm


class br2MPC:
    def __init__(self, N=20) -> None:
    
        self.N  = N
        self.DT = 0.01

        self.vx_max = 1 
        self.vy_max = 1 
        self.vz_max = 1
        self.v_roll_max = np.pi/5
        self.v_pitch_max = np.pi/5 
        self.v_yaw_max = np.pi/5
        self.tau_x_max = 50
        self.tau_y_max = 50
        self.tau_z_max = 10
        self.tau_roll_max = 10
        self.tau_pitch_max = 10
        self.tau_yaw_max = 10



        x = ca.SX.sym('x') 
        y = ca.SX.sym('y')
        z = ca.SX.sym('z')
        roll = ca.SX.sym('roll')
        pitch = ca.SX.sym('pitch')
        yaw = ca.SX.sym('yaw')
        v_x = ca.SX.sym('v_x')
        v_y = ca.SX.sym('v_y')
        v_z = ca.SX.sym('v_z')
        v_roll = ca.SX.sym('v_roll')
        v_pitch = ca.SX.sym('v_pitch')
        v_yaw = ca.SX.sym('v_yaw')
        
        self.states = ca.vertcat(x,y,z,roll,pitch,yaw,v_x,v_y,v_z,v_roll,v_pitch,v_yaw)
        self.n_states = self.states.size()[0]
        self.n_goal = 6

        tau_x = ca.SX.sym('tau_x')
        tau_y = ca.SX.sym('tau_y')
        tau_z = ca.SX.sym('tau_z')
        tau_roll = ca.SX.sym('tau_roll')
        tau_pitch = ca.SX.sym('tau_pitch')
        tau_yaw = ca.SX.sym('tau_yaw')
        
        self.controls = ca.vertcat(tau_x,tau_y,tau_z,tau_roll,tau_pitch,tau_yaw)
        self.n_controls = self.controls.size()[0]

        m,W,B,Ix,Iy,Iz,Xu_d,Yv_d,Zw_d,Kp_d,Mq_d,Nr_d,Xu,Yv,Zw,Kp,Mq,Nr,Xuu,Yvv,Zww,Kpp,Mqq,Nrr,rg =11.5, 112.8, 114.8, 0.16, 0.16, 0.16, -5.5, -12.7, -14.57, -0.12, -0.12, -0.12, -4.03, -6.22, -5.18, -0.07, -0.07, -0.07, -18.18, -21.66, -36.99, -1.55, -1.55,-1.55, 0.02

        M = ca.DM([ [m - Xu_d,  0,          0,          0,          m * rg,         0           ],
                    [0,         m - Yv_d,   0,         -m * rg,     0,              0           ],
                    [0,         0,          m - Zw_d,   0,          0,              0           ],
                    [0,        -m * rg,     0,          Ix - Kp_d,  0,              0           ],
                    [m * rg,    0,          0,          0,          Iy - Mq_d,      0           ],
                    [0,         0,          0,          0,          0,              Iz - Nr_d   ]  ] )

        C = ca.vertcat(v_pitch * (Zw_d * v_z + m * v_z),
                        -v_roll * (Zw_d * v_z + m * v_z) - Xu_d * v_x * v_yaw,
                        v_pitch * (Xu_d * v_x - m * v_x) - v_roll * (Yv_d * v_y - m * v_y),
                        v_pitch * (Iz * v_yaw - Nr_d * v_yaw) - v_yaw * (Iy * v_pitch - Mq_d * v_pitch) + v_z * (Yv_d * v_y - m * v_y) - v_y * (Zw_d * v_z - m * v_z),
                        v_yaw * (Ix * v_roll - Kp_d * v_roll) - v_roll * (Iz * v_yaw - Nr_d * v_yaw) - v_z * (Xu_d * v_x + m * v_x) + v_x * (Zw_d * v_z - m * v_z),
                        v_roll * (Iy * v_pitch - Mq_d * v_pitch) - v_pitch * (Ix * v_roll - Kp_d * v_roll) + v_y * (Xu_d * v_x - m * v_x) - v_x * (Yv_d * v_y - m * v_y))

        D = ca.vertcat(-(Xu + (Xuu * ca.fabs(v_x))) * v_x,
                        -(Yv + (Yvv * ca.fabs(v_y))) * v_y,
                        -(Zw + (Zww * ca.fabs(v_z))) * v_z,
                        -(Kp + (Kpp * ca.fabs(v_roll))) * v_roll,
                        -(Mq + (Mqq * ca.fabs(v_pitch))) * v_pitch,
                        -(Nr + (Nrr * ca.fabs(v_yaw))) * v_yaw)

        gRF = ca.vertcat(    (W - B) * ca.sin(y),
                            -(W - B) * ca.cos(y) * ca.sin(x),
                            -(W - B) * ca.cos(y) * ca.cos(x),
                             rg * W * ca.cos(y) * ca.sin(x),
                            rg * W * ca.sin(y),
                            0 )                

        J_n = ca.vertcat(ca.horzcat(ca.cos(yaw) * ca.cos(pitch), ca.cos(yaw) * ca.sin(roll) * ca.sin(pitch) - ca.cos(roll) * ca.sin(yaw), ca.sin(roll) * ca.sin(yaw) + ca.cos(roll) * ca.cos(yaw) * ca.sin(pitch), 0, 0, 0),
                    ca.horzcat(ca.cos(pitch) * ca.sin(yaw), ca.cos(roll) * ca.cos(yaw) + ca.sin(roll) * ca.sin(yaw) * ca.sin(pitch), ca.cos(roll) * ca.sin(yaw) * ca.sin(pitch) - ca.cos(yaw) * ca.sin(roll), 0, 0, 0),
                    ca.horzcat(-ca.sin(pitch), ca.cos(pitch) * ca.sin(roll), ca.cos(roll) * ca.cos(pitch), 0, 0, 0),
                    ca.horzcat(0, 0, 0, 1, ca.sin(roll) * ca.tan(pitch), ca.cos(roll) * ca.tan(pitch)),
                    ca.horzcat(0, 0, 0, 0, ca.cos(roll), -ca.sin(roll)),
                    ca.horzcat(0, 0, 0, 0, ca.sin(roll) / ca.cos(pitch), ca.cos(roll) / ca.cos(pitch)))
                        
        ## rhs
        rhs = ca.vertcat(ca.mtimes(J_n, self.states[6:12]), ca.mtimes(ca.inv(M), self.controls - C - D - gRF))

        ## function
        f = ca.Function('f', [self.states, self.controls], [rhs])

        ## for MPC
        T   = ca.SX.sym('T', self.n_controls, self.N )          # T (controls)
        S   = ca.SX.sym('S', self.n_states, self.N+1 )          # S (states)
        P   = ca.SX.sym('P', self.n_states + self.n_goal )      # P (parameter)

        ### define
        #Q = ca.diagcat(70,70,70,30,30,30,1,1,1,1,1,1)
        Qp = ca.diagcat(10,10,1,1,1,10)                 # Position error weights
        #Qv = ca.diagcat(15,15,15,100,100,100)           # Velocity error weights
        Qv = ca.diagcat(1,1,1,1,1,1)                    # Velocity error weights

        q = 10000
        Q = ca.diagcat(q,q,q,q,q,q)                     # CLF weights

        R = ca.diagcat(1,1,1,1,1,1)                     # Control Weights
        V = 10**7*ca.diagcat(1,1,1,1,1,5)

        ## Cost Function
        obj = 0 # initalise objective function
        g = [] # initalise constraints vector
        
        g.append(S[0:12, 0]- P[0:12, 0])
                
        for i in range(self.N):

            psn_err = S[0:6, i] - P[12:, 0]
            vel_err = S[6:12,i] - P[6:12,0]

            obj = obj + ca.mtimes( [psn_err.T, Qp, psn_err] ) + ca.mtimes( [vel_err.T, Qv, vel_err] ) + ca.mtimes([T[:,i].T, R, T[:,i]])
            
            # RK4
            # k1 = f(S[:, i], T[:, i])
            # k2 = f(S[:, i] + self.DT/2*k1, T[:, i])
            # k3 = f(S[:, i] + self.DT/2*k2, T[:, i])
            # k4 = f(S[:, i] + self.DT*k3, T[:, i])
            # stNext = 

            # Euler Method
            x_next_ = f(S[:, i], T[:, i])*self.DT + S[:, i]
            g.append(S[:, i+1] - x_next_ )   
        
        
        state_error_N = S[0:6, self.N] - P[12:, 0]    
        obj = obj + ca.mtimes([state_error_N[0:6].T, V, state_error_N[0:6]])
                
        opt_variables = ca.vertcat( ca.reshape(T, -1, 1), ca.reshape(S, -1, 1))
        opt_params = ca.reshape(P, -1, 1)

        nlp_prob = {'f': obj, 'x': opt_variables, 'p':opt_params, 'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level': 0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        print(nlp_prob)
        print(self.solver)

        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []


        for _ in range(self.n_states * (self.N+1) ):
            self.lbg.append(0.0)
            self.ubg.append(0.0)

        # for _ in range(N*n_SO):
        #     lbg.append(0.0)
        #     ubg.append(np.inf)   
         
        for _ in range(self.N):  # boundary of control input
            self.lbx += [-self.tau_x_max, -self.tau_y_max, -self.tau_z_max, -self.tau_roll_max, -self.tau_pitch_max, -self.tau_yaw_max]
            self.ubx += [ self.tau_x_max,  self.tau_y_max,  self.tau_z_max,  self.tau_roll_max,  self.tau_pitch_max,  self.tau_yaw_max]
        for _ in range(self.N+1): # boundary of state
            self.lbx += [ -100, -100, -2, -np.inf, -np.inf, -np.inf, -self.vx_max, -self.vy_max, -self.vz_max, -self.v_roll_max, -self.v_pitch_max, -self.v_yaw_max]
            self.ubx += [  100,  100,  2,  np.inf,  np.inf,  np.inf,  self.vx_max,  self.vy_max,  self.vz_max,  self.v_roll_max,  self.v_pitch_max,  self.v_yaw_max]





if __name__ == '__main__':

    # >>>>>>>> GLOBAL VARIABLES <<<<<<<<<<

    # ROS
    rov_id = 1                                      # set rov position in formation
    rnodeName = "rov" + str(rov_id)                 # set node name from rovid
    rPosSub = "/rov" + str(rov_id-1) + "_position"  # set the neighbour rov to track
    rPosPub = "/rov" + str(rov_id) + "_position"    # set the publisher for this rov position

    # rospy.init_node(rnodeName, anonymous=False)             # setup ros node
    # rospy.Subscriber(rPosSub, NavSatFix, listenCB)          # setup subscriber to get neighbour position
    # rospy.Subscriber(rPosPub, Vector3, getFormation)        # subscriber for formation configuration
    # pub = rospy.Publisher('/rov1_position', NavSatFix, queue_size=1 ) # setup publisher for this rov position

    # LOCATION
    dlat =  []                  # global variable for desired latitude
    dlon =  []                  # global variable for desired longitude
    dyaw =  0                   # global variable for desired yaw
    flag_subscriber = False     # flag indicates if the subscriber is active (dont start until it is)

    osetN = 0                   # Latitude (North) offset for rov1
    osetE = 0                   # Longitude (East) offset for rov1

    lat0  =  54.558800          # Starting latitude
    lng0  = -06.451004          # Starting longitude

    # MAVLINK
    mavport = 14560
    mavaddress = 'udpin:0.0.0.0:' + str(mavport)
    master = mavutil.mavlink_connection(mavaddress)
    master.wait_heartbeat()

    # sequence control
    fleg1 = False
    fleg2 = False
    fleg3 = False
    fleg4 = True   # first time mpc solver is called go False, changes input parameters to solver

    t_last_ahrs2 = -999 # holds timestamp of previous reading, for calculating rates

    #mpc
    tau = np.array([0,0,0,0,0,0])

    N = 20
    rov = br2MPC(N)    

    #### START OF CONTROL LOOP ####
    # while not flag_subscriber:
    #     rospy.sleep(1)
    #     print("Waiting for subscriber...")

    # print("Subscriber Active")

    # sys_arm()

    while not rospy.is_shutdown():
        
        mpcFleg = fleg1 and fleg2 and fleg3
        
        if mpcFleg :

            J = np.array([[np.cos(rov_yaw), -np.sin(rov_yaw)] , [np.sin(rov_yaw), np.cos(rov_yaw)]])
            j = np.linalg.inv(J)
            evel = np.array([ [earth_vx], [earth_vy] ])
            rvel = j @ evel
            rov_vel_x = rvel[0,0]
            rov_vel_y = rvel[1,0]
            rov_vel_z = 0

            rovState    = np.array([rov_x, rov_y, rov_z, rov_pitch, rov_roll, rov_yaw, rov_vel_x, rov_vel_y, rov_vel_z, rov_pitch_vel, rov_roll_vel, rov_yaw_vel ])
            
            
            ##############################################################################################################
            
            goalState   = np.array([ 20.00, 0.00 ,-0.50, 0.00, 0.00, 0.00 ]); 

            ##############################################################################################################

            # yawErr = wraptopi(rovState[5] - goalState[5])

            # rovState[5] = yawErr * 10
            # goalState[5] = 0.00

            rovState    = rovState.reshape((-1,1))
            goalState   = goalState.reshape((-1,1))

            if fleg4:   # different argX0 settings for first mpc call        
                X0 = np.tile(rovState,(N+1, 1))
                u0 = np.tile(tau, (N, 1)) ; u0 = u0.reshape((-1,1))
                fleg4 = False

            else:
                X0 = np.concatenate( [rovState, pos_est])       # [ fb_state, state_est_t2, stest_t3, stest_t4, ... , sest_tN ]  : ( 12(N+1) x 1 )
                u0 = np.concatenate([ u_est, u_est[18:24] ])    # [ u_est_t2, u_est_t3 , ... , u_est(N-1), u_est_tN, u_est_tN]   : (      6N x 1 )

            # print(np.shape(u0))
            # print(np.shape(X0))
            arg_X0 = np.vstack((u0,X0))
            arg_p = np.vstack((rovState,goalState))

            sol = rov.solver(x0=arg_X0, p=arg_p, lbg=rov.lbg, lbx=rov.lbx, ubg=rov.ubg, ubx=rov.ubx)
            estimated_opt = sol['x'].full()
            u_now = estimated_opt[0:6]          # control signal for this timestep
            u_est = estimated_opt[6:(6*N)]      # used to generate u0 for mpc in next timestep

            #pos_est = np.concatenate([ estimated_opt[(6*N + 12):] , estimated_opt[-12:] ])    # used to generate X0 for mpc in next timestep
            pos_est = estimated_opt[(6*N + 12):]
            u_pwm = tau2pwm(u_now)

            # idxs = np.arange(0,np.size(pos_est,0),12)

            # x_est = pos_est[idxs]
            # print(x_est)

            #print(u_now)
            print('\n', rovState[0:3])
            # print("################")
            #print(u_pwm)

            # rosmsg = Vector3()                             # update the ros topic
            # rosmsg.x    = rov_x
            # rosmsg.y    = rov_y
            
            # pub.publish(rosmsg)

            # #des_x, des_y = local_pos(dlat,dlon)

            # des_y = dlon + osetE
            # des_x = dlat + osetN

            set_rc_channel_pwm(5,int(u_pwm[0])) # set surge control
            set_rc_channel_pwm(6,int(u_pwm[1])) # set lateral control
            #set_rc_channel_pwm(4,int(u_pwm[5])) # set yaw control

            tau = u_now


            fleg1 = False
            fleg2 = False
            fleg3 = False
        
        msg = master.recv_match()
        if not msg:
            continue

        if msg.get_type() == "GLOBAL_POSITION_INT":
            rov_lat = msg.lat / 10000000
            rov_lon = msg.lon / 10000000
            rov_yaw = msg.hdg * 0.01

            rov_yaw = wraptopi(np.deg2rad(rov_yaw))
            
            fleg1 = True


        if msg.get_type() == "LOCAL_POSITION_NED":

            rov_x  = msg.x
            rov_y  = msg.y
            rov_z  = msg.z
            earth_vx = msg.vx
            earth_vy = msg.vy
            # earth_vz = msg.vz

            fleg2 = True


        if msg.get_type() == "AHRS2": 
                       
                if t_last_ahrs2 != -999:
                    t_now_ahrs2 = time.time()
                    dt = t_now_ahrs2 - t_last_ahrs2
                    dyaw = rov_yaw - msg.yaw
                    droll = rov_roll - msg.roll
                    dpitch = rov_pitch - msg.pitch
                    rov_yaw_vel = dyaw / dt
                    rov_roll_vel = droll / dt
                    rov_pitch_vel = dpitch / dt
                    # print("ROV Yaw Rate : " , f'{rov_yaw_vel:07.3f}', " Roll : ", f'{rov_roll_vel:07.3f}', " Pitch : ", f'{rov_pitch_vel:07.3f}')
                    t_last_ahrs2 = t_now_ahrs2

                    fleg3 = True

                else:
                    
                    rov_yaw_vel = 0.00
                    rov_roll_vel = 0.00
                    rov_pitch_vel = 0.00
                    t_last_ahrs2 = time.time()

                rov_yaw     = msg.yaw
                rov_pitch   = msg.pitch
                rov_roll    = msg.roll
                












    
    
