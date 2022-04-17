# P[][]是环境定义的变量,存储状态s下采取动作a得到的元组数据（转移概率，下一步状态，奖励，完成标志）
# next_state是否是终止状态？if Yes：done=True；else：done=False
def get_xianggehuanjing():  #我将环境修改了一下

    P = {s : {a : [] for a in range(4)} for s in range(16)}


    P[0][0].append((float(1),0,float(0),False))
    P[0][1].append((float(1),4,float(0),False))
    P[0][2].append((float(1),1,float(0),False))
    P[0][3].append((float(1),0,float(0),False))

    P[1][0].append((float(1),0,float(0),False))
    P[1][1].append((float(1),5,float(0),True))
    P[1][2].append((float(1),2,float(0),False))
    P[1][3].append((float(1),1,float(0),False))

    P[2][0].append((float(1),1,float(0),False))
    P[2][1].append((float(1),6,float(0),False))
    P[2][2].append((float(1),3,float(0),False))
    P[2][3].append((float(1),2,float(0),False))

    P[3][0].append((float(1),2,float(0),False))
    P[3][1].append((float(1),7,float(0),False))
    P[3][2].append((float(1),3,float(0),False))
    P[3][3].append((float(1),3,float(0),False))


    P[4][0].append((float(1),4,float(0),False))
    P[4][1].append((float(1),8,float(0),False))
    P[4][2].append((float(1),5,float(0),True))
    P[4][3].append((float(1),0,float(0),False))


    P[5][0].append((float(1),5,float(0),True))
    P[5][1].append((float(1),5,float(0),True))
    P[5][2].append((float(1),5,float(0),True))
    P[5][3].append((float(1),5,float(0),True))


    P[6][0].append((float(1),5,float(0),True))
    P[6][1].append((float(1),10,float(0),False))
    P[6][2].append((float(1),7,float(0),True))
    P[6][3].append((float(1),2,float(0),False))


    P[7][0].append((float(1),7,float(0),True))
    P[7][1].append((float(1),7,float(0),True))
    P[7][2].append((float(1),7,float(0),True))
    P[7][3].append((float(1),7,float(0),True))


    P[8][0].append((float(1),8,float(0),False))
    P[8][1].append((float(1),12,float(0),True))
    P[8][2].append((float(1),9,float(0),False))
    P[8][3].append((float(1),4,float(0),False))


    P[9][0].append((float(1),8,float(0),False))
    P[9][1].append((float(1),13,float(0),False))
    P[9][2].append((float(1),10,float(0),False))
    P[9][3].append((float(1),5,float(0),True))


    P[10][0].append((float(1),9,float(0),False))
    P[10][1].append((float(1),14,float(0),False))
    P[10][2].append((float(1),11,float(0),True))
    P[10][3].append((float(1),6,float(0),False))


    P[11][0].append((float(1),11,float(0),True))
    P[11][1].append((float(1),11,float(0),True))
    P[11][2].append((float(1),11,float(0),True))
    P[11][3].append((float(1),11,float(0),True))


    P[12][0].append((float(1),12,float(0),True))
    P[12][1].append((float(1),12,float(0),True))
    P[12][2].append((float(1),12,float(0),True))
    P[12][3].append((float(1),12,float(0),True))


    P[13][0].append((float(1),12,float(0),False))
    P[13][1].append((float(1),13,float(0),False))
    P[13][2].append((float(1),14,float(0),False))
    P[13][3].append((float(1),9,float(0),False))


    P[14][0].append((float(1),13,float(0),False))
    P[14][1].append((float(1),14,float(0),False))
    P[14][2].append((float(1),15,float(1),True))
    P[14][3].append((float(1),10,float(0),False))


    P[15][0].append((float(1),15,float(0),True))
    P[15][1].append((float(1),15,float(0),True))
    P[15][2].append((float(1),15,float(0),True))
    P[15][3].append((float(1),15,float(0),True))

    return P


