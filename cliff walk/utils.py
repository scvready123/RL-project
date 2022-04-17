#为了使得agent尽量走safer path,可以将最上面一行的网格的reward修改为比-1大的值
new_reward=-0.5#-0.5似乎是一个比较好的值，某些值会让收敛变得很慢。

def change_env(A):    #修改环境参数

    A[0] = {0: [(1.0, 0,new_reward, False)], 1: [(1.0, 1, new_reward, False)], 2: [(1.0, 12, -1, False)], 3: [(1.0, 0, new_reward, False)]}
    A[11] = {0: [(1.0, 11, new_reward, False)], 1: [(1.0, 11, new_reward, False)], 2: [(1.0, 23, -1, False)], 3: [(1.0, 10, new_reward, False)]}
    A[12] = {0: [(1.0, 0, new_reward, False)], 1: [(1.0, 13, -1, False)], 2: [(1.0, 24, -1, False)], 3: [(1.0, 12, -1, False)]}
    A[23] = {0: [(1.0, 11, new_reward, False)], 1: [(1.0,23, -1, False)], 2: [(1.0, 35, -1, False)], 3: [(1.0, 22, -1, False)]}
    for i in range(10):
        A[i+1] = {0: [(1.0, i+1, new_reward, False)], 1: [(1.0, i+2, new_reward, False)], 2: [(1.0, i+13, -1, False)], 3: [(1.0, i, new_reward, False)]}
    return A
