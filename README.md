RL课程项目

frozen lake: 这个文件夹有三个文件:main.py就是在gym自带的frozen-lake环境中采用价值迭代求解该问题。another-try.py就是在此基础上对环境做一些调整，会得到一些新的结果。utils.py封装了一些函数。

blackjack： 这个文件的内容如下：off-policy-TD.py,使用TD(1)方法，在blackjack环境中进行off-policy学习。on-policy-MC.py,使用蒙特卡洛方法，在blackjack环境中进行on-policy学习。result_picture存放将学习结果可视化的图片。 注：对于最优策略热力图的解释：其中横轴为庄家明的牌（从左到右为A-10）,纵轴为玩家当前牌点数之和（从下到上为12-21）,橙色格点为要牌，黄色格点为停牌。对于状态函数三维图的解释：Usable_Ace和NO_Usable_Ace分别为手中有可用A和无可用A的情况。

cliff walk: 这个文件夹的内容如下：Q-learning.py,使用Q-learning方法对智能体进行训练。Sarsa.py，使用Sarsa方法对智能体进行训练。utils.py里面封装了一些函数。文件夹Q-learing,Sarsa分别存放了两个方法的Q-table。 注：对于第4个问题的回答：只需要修改环境中某些格点的reward值，就可以使得agent更加倾向于走safer path,这点在utils.py中实现。
