'''
由于gym中自带的FrozenLake-v1环境中agent进行的是不确定性转移
所以迭代最后得到的策略似乎有点奇怪（直观感受）,当然结果还是没有问题的
故我将环境做了一些修改，将agent每步的转移都视为确定性转移，并通过调整折扣因子的值(gamma=0.9得到了新的结果
'''

import gym
import numpy as np
from utils import get_xianggehuanjing

env = gym.make('FrozenLake-v0')
eon = env.observation_space.n
ean = env.action_space.n
J=get_xianggehuanjing()

#采用价值迭代
def value_itration(env, gamma):
    # 初始化状态值表（V表）
    value_table = np.zeros(eon)
    # 迭代次数
    no_of_iterations = 100000
    # 收敛判断阈值
    threshold = 1e-20
    # 开始迭代
    for i in range(no_of_iterations):
        # 初始化更新后的V表（旧表复制过来）
        updated_value_table = np.copy(value_table)
        # 计算每个状态下所有行为的next_state_rewards,并更新状态动作值表（Q表），最后取最大Q值更新V表
        # 遍历每个状态
        for state in range(eon):  #eon=16,对应表格里面16个状态
            # 初始化存储Q值的列表
            Q_value = []
            # 遍历每个动作
            for action in range(ean):   #ean=4,有4个action
                # 初始化存储下一个状态的奖励的列表
                next_states_rewards = []
                # P[][]是环境定义的变量,存储状态s下采取动作a得到的元组数据（转移概率，下一步状态，奖励，完成标志）
                for next_sr in J[state][action]:
                    # next_state是否是终止状态？if Yes：done=True；else：done=False
                    trans_prob, next_state, reward, done = next_sr
                    # 计算next_states_reward（公式）
                    next_states_rewards.append(
                        (trans_prob*(reward+gamma*updated_value_table[next_state])))
                    # 计算Q值（公式）
                    Q_value.append(np.sum(next_states_rewards))
                    # 取最大Q值更新V表，即更新当前状态的V值
                    value_table[state] = max(Q_value)

        # 收敛判断
        if(np.sum(np.fabs(updated_value_table-value_table)) <= threshold):
            print("在第%d次迭代后收敛" % (i+1))
            break
    if i<=10:
        print('收敛得很快啊')

    # 返回V表
    return value_table

# 策略选取
def extract_policy(value_table, gamma):
    # 初始化存储策略的数组
    policy = np.zeros(eon)
    # 对每个状态构建Q表，并在该状态下对每个行为计算Q值，
    for state in range(eon):
        # 初始化Q表
        Q_table = np.zeros(ean)
        # 对每个动作计算
        for action in range(ean):
            # 同上
            for next_sr in J[state][action]:
                trans_prob, next_state, reward, done = next_sr
                # 更新Q表，即更新动作对应的Q值（4个动作分别由0-3表示）
                Q_table[action] += (trans_prob *
                                    (reward+gamma*value_table[next_state]))
        # 当前状态下，选取使Q值最大的那个动作
        policy[state] = np.argmax(Q_table)
    # 返回动作
    return policy

#折扣因子
gamma=0.9

# 最优值函数
optimal_value_function = value_itration(env=env, gamma=gamma)
# 最优策略
optimal_policy = extract_policy(optimal_value_function, gamma=gamma)

# 输出价值函数与最优策略
optimal_value_function=optimal_value_function.reshape(4,4)  #将输出转化成网格的形式更加的直观
optimal_policy=optimal_policy.reshape(4,4)   #将输出转化成网格的形式更加的直观

print('价值函数示意图:')
print(optimal_value_function)
print('最优策略示意图:')
print(optimal_policy)