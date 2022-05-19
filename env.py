import numpy as np
import random


# assumptions:
# number of users and occupied bandwidth is one to one
# SSI is constant
# Handover overhead is constant

# k = 7
# neighbors:
# f1: {f2,f4,f5}
# f2: {1,5,4}
# f3: {2,5,6}
# f4: {1,5,7}
# f5: {1,2,3,4,6,7}
# f6: {3,5,7}
# f7: {4,5,6}

class RANEnv():
    def __init__(self):
        self.mycounter = 0
        self.cluster={
            0:[],
            1:[],
            2:[],
            3:[],
            4:[],
            5:[],
            6:[]
        }
        self.FNodes = {
            0 : [1,0,5,6],
            1 : [0,1,2,6],
            2 : [2,3,1,6],
            3 : [2,3,4,6],
            4 : [4,5,6,3],
            5 : [0,4,5,6],
            6 : [0,1,2,3,4,5,6],
            7 : [0]
        }
        self.environment_2 = {
            1: 0.012,
            2: 0.058,
            3: 0.288,
            4: 0.230,
            5: 0.162,
            6: 0.071,
            7: 0.064,
            8: 0.057,
            9: 0.043,
            10: 0.015
        }
        self.environment_1 = {
            1: 0.015,
            2: 0.073,
            3: 0.365,
            4: 0.292,
            5: 0.205,
            6: 0.014,
            7: 0.013,
            8: 0.011,
            9: 0.009,
            10: 0.003
        }
        self.environment = {
            1: 0.008,
            2: 0.038,
            3: 0.192,
            4: 0.154,
            5: 0.108,
            6: 0.142,
            7: 0.129,
            8: 0.114,
            9: 0.086,
            10: 0.029
        }

        # k = number of fog nodes
        # N = resource capacity of each Fnode
        # C = set of possible resource blocks
        # H = set of possible holding times
        self.f_b=np.zeros(7)
        self.f_l=np.zeros(7)
        self.k = 7
        self.N = 7
        self.C = [1,2,3,4]          #0.1,0.2,0.3,04
        self.H = [5,10.15,20,25,30] #0.05,0.1,0.1,0.15,0.2
        self.u_h = 8

        self.c_t = random.choices(self.C, weights=(0.1, 0.2, 0.3, 0.4), k=1)[0]
        self.h_t = random.choices(self.H, weights=(0.05, 0.1, 0.1, 0.15, 0.2), k=1)[0]
        self.u_t = random.choices(list(self.environment.keys()),list(self.environment.values()),k=1)[0]
        self.u_h = 8
        self.f_hat = random.randint(0,self.k-1)

        self.state=[self.f_hat,self.u_t,self.c_t,self.h_t]
        for i in range(0,self.k):
            self.f_b[i]=self.N
            self.f_l[i]=max(self.C)*max(self.H)
            self.state= self.state + [self.f_b[i],self.f_l[i]]


        # selections that UE can have: 3 BS and 2 NS, {(1,1),(1,2),(2,2),(3,2)}
        self.action_space = np.zeros(self.k+1)

        # self.observation_space =
        # Bandwidth array
        #         self.observation_space= Box(low=np.array([0]), high=np.array([10]))
        # set the start bandwidth
        #         self.state = 5+random.randint(-4, 4)
        # set decision_time
        self.situation_reward = {
            "r_sh": 24,
            "r_sl": -12,
            "r_rh": -12,
            "r_rl": -3,
            "r_bh": 3,
            "r_bl": 12,
            "r_illegal": -200}

        self.actions = [0]
        for i in range(1, self.k+1):
            self.actions.append(i)

    def calc_reward(self, action):
        reward = 0
        _used_blocks=0
        _illegal_action = False

        for key in list(self.cluster.keys()):
            if key==action:
                _used_blocks=sum(i for i, _ in self.cluster[key])

                if action not in self.FNodes[action]:
                    reward += self.situation_reward["r_illegal"]  # next time quit the game because of taking an illegal action
                    # print("selected a wrong number")
                    _illegal_action = True

                elif self.N-_used_blocks<self.c_t:
                    reward +=self.situation_reward["r_illegal"]
                    # print("selected node didnt have enough space.")
                    _illegal_action = True

            if _illegal_action:
                return reward,_illegal_action

        if action != self.k:
            if self.state[1] >= self.u_h:reward = reward + self.situation_reward["r_sh"]
            if self.state[1] < self.u_h:reward = reward + self.situation_reward["r_sl"]

        if action == self.k:
            if self.state[1] >= self.u_h: reward = reward + self.situation_reward["r_rh"]
            if self.state[1] < self.u_h: reward = reward + self.situation_reward["r_rl"]
        #TODO add the condition for r_bh, and r_bl, rejecting because of being busy

        reward_l=max(self.C)*max(self.H)+1-self.state[2]*self.state[3]
        if action == self.k :
            reward = reward - reward_l
        else:
            reward = reward+reward_l
        return reward,_illegal_action

    def step(self, action,timer):
        done = False
        reward, illegal_action = self.calc_reward(action)

        updated_cluster=self.update_state(action,illegal_action,timer)
        self.c_t = random.choices(self.C, weights=(0.1, 0.2, 0.3, 0.4), k=1)[0]
        self.h_t = random.choices(self.H, weights=(0.05, 0.1, 0.1, 0.15, 0.2), k=1)[0]
        self.u_t = random.choices(list(self.environment.keys()), list(self.environment.values()), k=1)[0]
        self.u_h = 8
        self.f_hat = random.randrange(1, self.k)
        self.state=[self.f_hat,self.u_t,self.c_t,self.u_h]+updated_cluster

        return self.state, reward, done

    def update_state(self,action,illegal_action,timer):
        # find the Fog node and update the tuples with self.c_t and self.expire time
        _self_update=[]
        for key in list(self.cluster.keys()):
            if key == action and not illegal_action:
                self.cluster[key].append((self.c_t , self.h_t+timer))
            for (x,y) in self.cluster[key]:
                if y <= timer:
                    self.cluster[key].remove((x,y))
                # print(self.cluster)
                self.f_b[key]=self.N-sum(i for i, _ in self.cluster[key])
                self.f_l[key]=sum(i*j for (i,j) in self.cluster[key])
            _self_update=_self_update + [self.f_b[key],self.f_l[key]]

        #outside of if, find if any expire time is equal to timer, remove that tuple
        return _self_update

        # self.c_t = random.choices(self.C, weights=(0.1, 0.2, 0.3, 0.4), k=1)[0]
        # self.h_t = random.choices(self.H, weights=(0.05, 0.1, 0.1, 0.15, 0.2), k=1)[0]
        # self.u_t = random.choices(list(self.environment.keys()), list(self.environment.values()), k=1)[0]
        # self.u_h = 8
        # self.f_hat = random.randrange(1, self.k)
        #
        # self.expired_time=self.h_t+self.timer
        #
        #
        # self.state = [self.f_hat, self.u_t, self.c_t, self.h_t]
        # for i in range(0, self.k):
        #     self.f_b[i] = self.N
        #     self.f_l[i] = max(self.C) * max(self.H)
        #     self.state = self.state + [self.f_b[i], self.f_l[i]]

    def render(self):
        pass

    def reset(self):
        self.c_t = random.choices(self.C, weights=(0.1, 0.2, 0.3, 0.4), k=1)[0]
        self.h_t = random.choices(self.H, weights=(0.05, 0.1, 0.1, 0.15, 0.2), k=1)[0]
        self.u_t = random.choices(list(self.environment.keys()), list(self.environment.values()), k=1)[0]
        self.u_h = 8
        self.f_hat = random.randrange(1, self.k)

        self.state = [self.f_hat, self.u_t, self.c_t, self.h_t]
        for i in range(0, self.k):
            self.f_b[i] = self.N
            self.f_l[i] = max(self.C) * max(self.H)
            self.state = self.state + [self.f_b[i], self.f_l[i]]
        self.mycounter +=1


        return self.state





