import numpy as np

# Q Table
Q = np.zeros([7,2])
# Learning Rate
lr = .9
# Future Reward value
y = .9
# Number of runs
num_runs = 100
# List containing reward we achieved on each run
rList = []

# What our game world looks like
# -1 Loss
# 1 Win
# 0 Open Space
gameMap = [-1,0,0,0,1,0,-1]

# Train Q-Table num_runs times
for i in range(num_runs):
  alive = True
  won = False

  rAll = 0
  j = 0
  # Player current state at a random open position
  s = np.random.choice([1,2,3,5])

  # Make a maximum of 10 actions to prevent infinite loop
  while(alive and won != True and j < 10):
    j += 1
    # choose action
    print("S"+str(s)+":"+str(Q[s,:]))
    a = np.argmax(Q[s,:] + np.random.randn(1, 2)*(1./(i+1)))
    # get new state and reward
    if(a == 0):
      print("Move Left")
      sl = s-1
    else:
      print("Move Right")
      sl = s+1
    # reward is equal to value of new position on map
    r = gameMap[sl]
    # update
    # Reward added to
    # Value of best action that we know at our new state
    # Multiplied by our discovery rate and learning rate
    Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[sl,:]) - Q[s,a])

    # set current state to the new state we just moved to
    s = sl

    # Reward tracking for performance analysis
    rAll += r

    #c Check if game has been won or lose
    if(r == 1):
      won = True
      print("Game Won")
    if(r == -1):
      alive = False
      print("Game Lost")

  # Add how we performed on that run to our history    
  rList.append(rAll)

# Print details
print("Score over time: " + str(sum(rList)/num_runs))
print("Final Q Table")
print("  LEFT         RIGHT")
print(Q)