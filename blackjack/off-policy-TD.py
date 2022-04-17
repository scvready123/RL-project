import matplotlib
import numpy as np
import gym
import sys
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 编写三维画图函数
def plot_3D(X, Y, Z, xlabel, ylabel, zlabel, title):
    fig = plt.figure(figsize=(20, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.rainbow, vmin=-1.0, vmax=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    ax.set_facecolor("white")
    fig.colorbar(surf)
    return fig


def OffPolicy(env, num_episodes,epsilon,alpha):
    """
    env         : 问题环境
    num_episodes: 幕数量
    return      : 返回状态价值函数与最优策略
    """


    # 初始化策略(任何状态下都不要牌)
    policy = defaultdict(int)
    # 初始化回报和
    r_sum = defaultdict(float)
    # 初始化访问次数
    r_count = defaultdict(float)
    # 初始化状态价值函数
    r_v = defaultdict(float)
    # 对各幕循环迭代
    epsilon_final = 0.0001  # 贪婪因子最终的值
    j = epsilon

    for each_episode in range(num_episodes):
        # 输出迭代过程
        print("Episode {}/{}".format(each_episode, num_episodes), end="\r")
        sys.stdout.flush()
        epsilon=j+(each_episode-1)*((j-epsilon_final)/(1-num_episodes))#epsilon线性衰减
        #print(j)
        #print('epsilon',epsilon)
        # 初始化空列表记录幕过程
        episode = []
        # 初始化环境
        state = env.reset()

        # 生成（采样）幕
        done = False
        while not done:
            # 根据当前状态获得策略下的下一动作
            action_list = np.random.choice([policy[state], 1 - policy[state]], 1, replace=False,
                                           p=[1 - epsilon / 2, epsilon / 2])  #贪婪选择
            action = action_list[0]
            #print(action_list)
            # 得到下一个状态、回报以及该幕是否结束标志
            next_state, reward, done, info = env.step(action)
            # 对幕进行采样并记录
            episode.append((state, action, reward))
            #print(episode)
            # 更新状态
            state = next_state
            #print(state)
        k=0
        if len(episode)>=2:
            for i in range(len(episode) - 1):
                state_visit1 = episode[i][0]
                each_action1 = episode[i][1]
                re = episode[i][2]
                state_visit2 = episode[i + 1][0]

                if r_v[(state_visit2, 0)] < r_v[(state_visit2, 1)]:
                    k = r_v[(state_visit2, 1)]
                else:
                    k = r_v[(state_visit2, 0)]

                r_v[(state_visit1, each_action1)] = r_v[(state_visit1, each_action1)] + alpha * (
                    re + k - r_v[(state_visit1, each_action1)])

                if r_v[(state_visit1, each_action1)] < r_v[(state_visit1, 1 - each_action1)]:
                    policy[state_visit1] = 1 - each_action1
        else:
            state_visit1 = episode[0][0]
            #print(state_visit1)
            each_action1 = episode[0][1]
            #print(each_action1)
            re = episode[0][2]
            #print(re)
            #print(r_v[(state_visit1, each_action1)])
            r_v[(state_visit1, each_action1)] = r_v[(state_visit1, each_action1)] + alpha * (
                re - r_v[(state_visit1, each_action1)])
            if r_v[(state_visit1, each_action1)] < r_v[(state_visit1, 1 - each_action1)]:
                policy[state_visit1] = 1 - each_action1   #贪婪选取动作


        #print(episode)
    return policy, r_v



def testpolicy(env, num_episodes,epsilon):
     """
         env         : 问题环境
         num_episodes: 幕数量
         return      : 返回状态价值函数与最优策略
     """
     a,b,c=0,0,0
     # 初始化策略(任何状态下都不要牌)
     # 对各幕循环迭代
     for each_episode in range(num_episodes):
         # 输出迭代过程
         print("Episode {}/{}".format(each_episode, num_episodes), end="\r")
         sys.stdout.flush()

         # 初始化空列表记录幕过程
         episode = []
         # 初始化环境
         state = env.reset()

         # 生成（采样）幕
         done = False
         while not done:
             # 根据当前状态获得策略下的下一动作
             action_list = np.random.choice([policy[state], 1 - policy[state]], 1, replace=False,
                                            p=[1 - epsilon / 2, epsilon / 2])  # 贪婪选择
             action = action_list[0]
             # print(action_list)
             # 驱动环境的物理引擎得到下一个状态、回报以及该幕是否结束标志
             next_state, reward, done, info = env.step(action)
             # 对幕进行采样并记录
             episode.append((state, action, reward))
             # print(episode)
             # 更新状态
             state = next_state
             # print(state)

         if episode[-1][2]==1:
             a=a+1
         elif episode[-1][2]==0:
             b=b+1
         else:
             c=c+1

     return a,b,c

# 处理价值矩阵方便后续绘图
def process_q_for_draw(q, policy, ace):
    """
    v     : 状态价值函数
    ace   : 是否有可用A
    return: 返回处理好的三个坐标轴
    """
    # 根据动作价值函数到处最优状态价值函数
    v = defaultdict(float)
    for state in policy.keys():
        v[state] = q[(state, policy[state])]
    # 生成网格点
    x_range = np.arange(12, 22)
    y_range = np.arange(1, 11)
    X, Y = np.meshgrid(x_range, y_range)

    # 根据是否有可用的A选择绘制不同的3D图
    if ace:
        Z = np.apply_along_axis(lambda _: v[(_[0], _[1], True)], 2, np.dstack([X, Y]))
    else:
        Z = np.apply_along_axis(lambda _: v[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    return X, Y, Z


# 处理策略方便后续绘图
def process_policy_for_draw(policy, ace):
    """
    policy:输入策略
    ace   :是否有可用A
    return:以二维数组形式返回
    """
    policy_list = np.zeros((10, 10))
    # 将字典形式换为列表，方便后续作图
    if ace:
        for playerscores in range(12, 22):
            for dealercard in range(1, 11):
                policy_list[playerscores - 12][dealercard - 1] = policy[(playerscores, dealercard, 1)]
    else:
        for playerscores in range(12, 22):
            for dealercard in range(1, 11):
                policy_list[playerscores - 12][dealercard - 1] = policy[(playerscores, dealercard, 0)]
    return policy_list


# 主函数
if __name__ == '__main__':
    # 从gym库中调用Blackjack-v1环境
    env = gym.make("Blackjack-v0")
    # 对策略进行评估（预测）
    policy, q = OffPolicy(env, num_episodes=5000000,epsilon=0.1,alpha=0.01)  #可以考虑把num_episodes设置大一些，效果较好，比如5000000
    #print(policy)
    #print(q)
    # 绘制最优策略矩阵热力图
    # 准备画布大小，并准备多个子图
    _, axes = plt.subplots(1, 2, figsize=(40, 20))
    # 调整子图的间距，wspace=0.1为水平间距，hspace=0.2为垂直间距
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # 这里将子图形成一个1*2的列表
    axes = axes.flatten()
    # 有可用ACE下的最优策略
    fig = sns.heatmap(np.flipud(process_policy_for_draw(policy, 1)), cmap="Wistia", ax=axes[0])
    fig.set_ylabel('Player Sum', fontsize=20)
    fig.set_yticks(list(reversed(range(10))))
    fig.set_xlabel('Dealer Open Card', fontsize=20)
    fig.set_xticks(range(10))
    fig.set_title('Usable Ace', fontsize=20)
    # 无可用ACE下的最优策略
    fig = sns.heatmap(np.flipud(process_policy_for_draw(policy, 0)), cmap="Wistia", ax=axes[-1])
    fig.set_ylabel('Player Sum', fontsize=20)
    fig.set_yticks(list(reversed(range(10))))
    fig.set_xlabel('Dealer Open Card', fontsize=20)
    fig.set_xticks(range(10))
    fig.set_title('NO Usable Ace', fontsize=20)
    plt.show()
    plt.savefig("./result_picture/TD/Optimal Policy.jpg")

    # 3D绘图-状态价值矩阵
    X, Y, Z = process_q_for_draw(q, policy, ace=True)
    fig = plot_3D(X, Y, Z, xlabel="Player Sum", ylabel="Dealer Open Card", zlabel="Value", title="Usable Ace")
    fig.show()
    fig.savefig("./result_picture/TD/Usable_Ace.jpg")
    X, Y, Z = process_q_for_draw(q, policy, ace=False)
    fig = plot_3D(X, Y, Z, xlabel="Player Sum", ylabel="Dealer Open Card", zlabel="Value", title="No Usable Ace")
    fig.show()
    fig.savefig("./result_picture/TD/NO_Usable_Ace.jpg")
    #测试学习到的策略
    a,b,c=testpolicy(env, num_episodes=5000,epsilon=0)#在测试的时候不进行贪婪选择
    print('进行5000轮测试')
    print('获胜',a,'局')
    print('平局',b,'局')
    print('失败',c,'局')