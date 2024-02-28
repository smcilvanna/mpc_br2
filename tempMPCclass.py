#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt

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
        self.tau_x_max = 30
        self.tau_x_max = 30
        self.tau_x_max = 30
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
        Qp = ca.diagcat(8,8,8,10,10,10)                 # Position error weights
        Qv = ca.diagcat(15,15,15,100,100,100)           # Velocity error weights

        q = 10000
        Q = ca.diagcat(q,q,q,q,q,q)                     # CLF weights

        R = ca.diagcat(1,1,1,1,1,1)                     # Control Weights
        V = 10**7*ca.diagcat(1,1,1,1,1,1)

        ## Cost Function
        obj = 0 # initalise objective function
        g = [] # initalise constraints vector
        
        g.append(S[0:6, 0]- P[12:, 0])
        g.append(S[6:12,0] - P[6:12,0])
        
        print(g)

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
        

        print(len(g))
        
        state_error_N = S[0:6, self.N] - P[12:, 0]    
        obj = obj + ca.mtimes([state_error_N[0:6].T, V, state_error_N[0:6]])
                
        opt_variables = ca.vertcat( ca.reshape(T, -1, 1), ca.reshape(S, -1, 1))
        opt_params = ca.reshape(P, -1, 1)

        nlp_prob = {'f': obj, 'x': opt_variables, 'p':opt_params, 'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []


        for _ in range(self.n_states *(self.N+1) ):
            self.lbg.append(0.0)
            self.ubg.append(0.0)

        print(np.shape(self.lbg))

        # for _ in range(N*n_SO):
        #     lbg.append(0.0)
        #     ubg.append(np.inf)   
         
        for _ in range(self.N):  # boundary of control input
            self.lbx += [-self.tau_x_max, -self.tau_x_max, -self.tau_x_max, -self.tau_roll_max, -self.tau_pitch_max, -self.tau_yaw_max]
            self.ubx += [ self.tau_x_max,  self.tau_x_max,  self.tau_x_max,  self.tau_roll_max,  self.tau_pitch_max,  self.tau_yaw_max]
        for _ in range(self.N+1): # boundary of state
            self.lbx += [ -100, -100, -2, -np.inf, -np.inf, -np.inf, -self.vx_max, -self.vy_max, -self.vz_max, -self.v_roll_max, -self.v_pitch_max, -self.v_yaw_max]
            self.ubx += [  100,  100,  2,  np.inf,  np.inf,  np.inf,  self.vx_max,  self.vy_max,  self.vz_max,  self.v_roll_max,  self.v_pitch_max,  self.v_yaw_max]


def t0setup(N):      
    rov = br2MPC(N)
    xgoal = np.array([15,2,0,0,0,0]); xgoal = xgoal.reshape((-1,1))
    x0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0]); x0 = x0.reshape((-1,1))
    X0 = np.tile(x0,(N+1, 1))
    X0 = np.reshape(X0,(-1,1))
    u0 = np.zeros((6,1))
    u0 = np.tile(u0, (N, 1)) ; u0 = u0.reshape(-1,1)
    arg_X0 = np.vstack((X0,u0))
    arg_p = np.vstack((x0,xgoal))
    return rov, arg_X0, arg_p

def tnextstep(N, pos_Opt, tau_Opt):

    x0 = np.reshape(pos_Opt[:,-2], ((-1,1))) # take position near end of predicition horizon as new start position
    # print(np.shape(nextState))

    xgoal = np.array([15,2,0,0,0,0]); xgoal = xgoal.reshape((-1,1))
    
    X0 = np.tile(x0,(N+1, 1))
    X0 = np.reshape(X0,(-1,1))
    
    u0 = np.reshape(tau_Opt[:,-2], ((-1,1)))    # take estimated control at end of prediciton as new start control
    u0 = np.tile(u0, (N, 1)) ; u0 = u0.reshape(-1,1)
    
    arg_X0 = np.vstack((X0,u0))
    arg_p = np.vstack((x0,xgoal))
    return arg_X0, arg_p




def plotrov(pos_Opt):
    xN = pos_Opt[0,:]
    yN = pos_Opt[1,:]
    zN = pos_Opt[2,:]
    pN = pos_Opt[3,:]
    rN = pos_Opt[4,:]
    wN = pos_Opt[5,:]

    xvN = pos_Opt[6,:]
    yvN = pos_Opt[7,:]
    zvN = pos_Opt[8,:]
    pvN = pos_Opt[9,:]
    rvN = pos_Opt[10,:]
    wvN = pos_Opt[11,:]

    fig, ax = plt.subplots()
    ax.scatter(xvN,yvN)
    plt.show()


if __name__ == '__main__':

    N = 15  # set prediction horizon
    rov, arg_X0, arg_p = t0setup(N) # call initial setup, create mpc solver + set initial values

    for tsteps in range(10):

        print("arg_X0  ", np.shape(arg_X0))
        print(arg_X0)
        print('\n')

        print("arg_p  ",np.shape(arg_p))
        print(arg_p)
        print('\n')


        print(np.shape(rov.lbg))
        print(np.shape(rov.ubg))

        sol = rov.solver(x0=arg_X0, p=arg_p, lbg=rov.lbg, lbx=rov.lbx, ubg=rov.ubg, ubx=rov.ubx)
        estimated_opt = sol['x'].full() 

        # print(estimated_opt,"\n")
        tau_Opt = estimated_opt[:int(rov.n_controls*rov.N)].reshape(rov.N, rov.n_controls).T        ## generatee n step control commands
        pos_Opt = estimated_opt[int(rov.n_controls*rov.N):].reshape(rov.N+1, rov.n_states).T        ## estimage n step pose states

        print("N control " , np.shape(tau_Opt))
        print(tau_Opt)
        print('\n')

        print("N position " , np.shape(pos_Opt))
        print(pos_Opt)
        print('\n')
        
        arg_X0, arg_p = tnextstep(N, pos_Opt, tau_Opt)
        

    # print(tau_Opt)
    # print(pos_Opt)
    plotrov(pos_Opt)




###////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# rov_id = 3      # set rov position in formation

# rnodeName = "rov" + str(rov_id)
# rPosSub = "/rov" + str(rov_id-1) + "_position"
# rPosPub = "/rov" + str(rov_id) + "_position"

# print("Node name : ", rnodeName)
# print("Subscriber : ", rPosSub)
# print("Publisher : ", rPosPub)

###////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# import numpy as np


# yaw = np.deg2rad(10)

# rov_vel = np.array([[2.4],[1.25]])

# J = np.array([[np.cos(yaw), -np.sin(yaw)] , [np.sin(yaw), np.cos(yaw)]])

# # print(J)
# # print(np.shape(J))

# earth_vel = J @ rov_vel

# print(earth_vel)

# j = np.linalg.inv(J)


# i_rov_vel = j @ earth_vel

# print(i_rov_vel)

