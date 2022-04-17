import gym
import numpy as np

#LEFT = 0
#DOWN = 1
#RIGHT = 2
#UP = 3

env = gym.make('FrozenLake-v0')
# 4*4的网格，有16个格子（状态），分别用0-15表示。eon=16
eon = env.observation_space.n
# 4个动作——上下左右，分别用0-3表示。ean=4
ean = env.action_space.n

# 价值迭代
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
                for next_sr in env.P[state][action]:
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
    # 返回V表
    return value_table


# 策略选取
def extract_policy(value_table, gamma):
    # 初始化存储策略的数组
    policy = np.zeros(eon)
    # 对每个状态构建Q表，并在该状态下对每个行为计算Q值，
    for state in range(eon):   #16个state
        # 初始化Q表
        #print('state',state)
        Q_table = np.zeros(ean)
        #print(Q_table)
        # 对每个动作计算
        for action in range(ean):   #每个state四个动作
            # 同上
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward, done = next_sr
                # 更新Q表，即更新动作对应的Q值（4个动作分别由0-3表示）
                Q_table[action] += (trans_prob *
                                    (reward+gamma*value_table[next_state]))
                #print('Q_table',Q_table)
        # 当前状态下，选取使Q值最大的那个动作
        policy[state] = np.argmax(Q_table)
    # 返回动作
    return policy

#折扣因子
gamma=1

# 最优值函数
optimal_value_function = value_itration(env=env, gamma=gamma)
# 最优策略
optimal_policy = extract_policy(optimal_value_function, gamma=gamma)

# 输出最优策略
optimal_value_function=optimal_value_function.reshape(4,4)  #弄成网格的形状，更加直观一些
optimal_policy=optimal_policy.reshape(4,4)  #弄成网格的形状，更加直观一些

print('价值函数示意图:')
print(optimal_value_function)
print('最优策略示意图:')
print(optimal_policy)