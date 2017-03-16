import numpy as np
import random

def tempdif(Q0,maxQ1,r,alpha,disc):
    return Q0 + alpha*(r + disc*maxQ1 - Q0)

board = [-1,0,0,1,0,-1]
win = 1
lose = -1

QTable = np.zeros([6,2])

print(QTable)

disc = 0.80
alpha = 0.85

num_runs = 100

for i in range(num_runs):
    ps = random.choice([1,2,4]) # start on nonterminals
    pa = 0 # no action
    while(True):
        na = np.argmax(QTable[ps,:] + np.random.randn(1,2)*(1.0/(i+1.0))) # next action
        if na == 0: # go left
            ns = ps - 1
        elif na == 1: # go right
            ns = ps + 1

        # update qtable for action with current state
        r = board[ns]
        QTable[ps,na] = tempdif(QTable[ps,na],np.max(QTable[ns,:]),r,alpha,disc)
        if r == 1:
            print("Won!")
            break
        elif r == -1:
            print("Lost!")
            break
        ps = ns
        pa = na
    print(QTable)
