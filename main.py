# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import dqn
from dqn import DQNAgent
from env import RANEnv
import numpy as np
import os
import matplotlib.pyplot as plt


def start():

    env = RANEnv()

    state_size = 18  # env.observation_space.shape[0]
    action_size = 8  # env.action_space.shape[0]
    number_of_fnodes = 7
    batch_size = 32
    n_episode = 1
    decision_time = 1
    agent = DQNAgent(state_size, action_size)
    #
    X = []
    Y = []

    output_dir = './model_output/NS'
    result_file = open("/home/a14154862/PycharmProjects/ns-paper/result_file.txt", "w")
    gos_file = open("/home/a14154862/PycharmProjects/ns-paper/gos_file.txt", "w")
    used_blocks_file = open("/home/a14154862/PycharmProjects/ns-paper/used_blocks_file.txt", "w")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    highly_served = 0
    total_highly = 0
    used_blocks = np.empty(8)
    used_blocks_txt = ""
    for e in range(n_episode):
        print(e)
        timer=0
        done = False
        total_reward = 0
        served_counter = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            action = agent.act(state)
            # print(action)
            next_state, reward, done= env.step(action,timer)


            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # print(state[0].shape)
            used_blocks_txt = used_blocks_txt + "{},{},{},{},{},{},{}".format(state[0][4],state[0][6],state[0][8],state[0][10],state[0][12],state[0][14],state[0][16])
            used_blocks_file.write(used_blocks_txt + "\n")
            used_blocks_txt = ""
            timer = timer + 1
            total_reward = total_reward + reward
            # np.savetxt(result_file, next_state, newline=" ")
            print(action)
            if timer > 30000:
                done = True
            else:
                done = False
            if state[0][1] > 7:  # 8 is threshold for high utility
                total_highly += 1
                if action < 7:  # 7 is number of Fog nodes so if the action is anything from 0 to 6 means it is served by
                    # cluster otherwise it is rejected.
                    highly_served += 1
            if action < 7:
                served_counter += 1
                # print("{} ----{}----- {}".format(action,decision_time,served_counter))
            if done or timer%100==0:
                print("episode:  {}/{}, score: {}, e: {:.2}".format(e, n_episode, total_reward, agent.epsilon))
                X.append(timer)
                Y.append(total_reward)
                result_file.write("{} , {}".format(state[0],  served_counter) + "\n")

                if total_highly != 0:
                    number = highly_served / total_highly
                else:
                    number = 0
                gos_file.write("{},{}".format(n_episode, number) + "\n")
        # if total_highly !=0:
        #     number = highly_served / total_highly
        # else:
        #     number=0
        # gos_file.write("{},{}".format(n_episode, number) + "\n")
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # if e % 50 == 0:
    agent.save()
    print("closing result_file")
    result_file.close()
    print("closing gos_file")
    gos_file.close()
    print("closing used_blocks")
    used_blocks_file.close()
    plt.plot(X, Y, marker="8")
    plt.show(block=False)
    # def print_hi(name):
    #     # Use a breakpoint in the code line below to debug your script.
    #     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    #
    #
    # # Press the green button in the gutter to run the script.
    # if __name__ == '__main__':
    #     print_hi('PyCharm')

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/


def validate():
    output_dir = './model_output/NS'

    action_1 =0 #remove
    action_2=0 #remove
    action_3=0
    action_4=0
    action_5=0
    action_6=0
    action_7=0
    action_0=0
    state_size = 18  # env.observation_space.shape[0]
    action_size = 8  # env.action_space.shape[0]
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    model.load_weights('./checkpoints/my_checkpoint')
    env = RANEnv()

    state_size = 18  # env.observation_space.shape[0]
    action_size = 8  # env.action_space.shape[0]
    number_of_fnodes = 7
    batch_size = 32
    n_episode = 1
    decision_time = 1
    agent = DQNAgent(state_size, action_size)
    #
    X = []
    Y = []

    result_file = open("/home/a14154862/PycharmProjects/ns-paper/result_file.txt", "w")
    gos_file = open("/home/a14154862/PycharmProjects/ns-paper/gos_file.txt", "w")
    used_blocks_file = open("/home/a14154862/PycharmProjects/ns-paper/used_blocks_file.txt", "w")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    highly_served = 0
    total_highly = 0
    used_blocks = np.empty(8)
    used_blocks_txt = ""
    for e in range(n_episode):

        timer = 0
        done = False
        total_reward = 0
        served_counter = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            print(state)
            act_values = model.predict(state)
            action = np.argmax(act_values[0])
            next_state, reward, done = env.step(action, timer)

            #remove
            if action ==0:
                action_0 +=1
            elif action ==1:
                action_1 +=1
            elif action ==2 :
                action_2+=1
            elif action ==3 :
                action_3+=1
            elif action == 4:
                action_4 += 1
            elif action ==5 :
                action_5+=1
            elif action ==6 :
                action_6+=1
            elif action ==7 :
                action_7+=1
            # end of remove

            state = next_state
            state = np.reshape(state, [1, state_size])
            used_blocks_txt = used_blocks_txt + "{},{},{},{},{},{},{}".format(state[0][4], state[0][6], state[0][8],
                                                                              state[0][10], state[0][12], state[0][14],
                                                                              state[0][16])
            used_blocks_file.write(used_blocks_txt + "\n")
            used_blocks_txt = ""
            timer = timer + 1
            total_reward = total_reward + reward
            # np.savetxt(result_file, next_state, newline=" ")

            if timer > 30000:
                done = True
            else:
                done = False
            if state[0][1] > 7:  # 8 is threshold for high utility
                total_highly += 1
                if action < 7:  # 7 is number of Fog nodes so if the action is anything from 0 to 6 means it is served by
                    # cluster otherwise it is rejected.
                    highly_served += 1
            if action < 7:
                served_counter += 1
                # print("{} ----{}----- {}".format(action,decision_time,served_counter))
            if done or timer % 100 == 0:
                # print("episode:  {}/{}, score: {}, e: {:.2}".format(e, n_episode, total_reward, agent.epsilon))
                X.append(timer)
                Y.append(total_reward)
                result_file.write("{} , {}".format(state[0], served_counter) + "\n")
                print(total_highly)
                print(highly_served)
                if total_highly != 0:
                    number = highly_served / total_highly
                else:
                    number = 0
                gos_file.write("{},{}".format(n_episode, number) + "\n")
                total_highly=0
                highly_served=0
    print("{},{}.{},{},{},{},{},{}".format(action_0,action_1,action_2,action_3,action_4,action_5,action_6,action_7))
    print("closing result_file")
    result_file.close()
    print("closing gos_file")
    gos_file.close()
    print("closing used_blocks")
    used_blocks_file.close()
    plt.plot(X, Y, marker="8")
    plt.show(block=False)

if __name__ == '__main__':
    # start()
    validate()
