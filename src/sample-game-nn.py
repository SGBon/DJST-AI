import numpy as np
import tensorflow as tf

# To avoid printing in scientific notation
np.set_printoptions(suppress=True)

# Set up a debug mode
debug = 1

# Learning Rate
lr = .5
# Future Reward value
y = .9

# Make a file to print to help see what's going on
filename = "nn_lr" + str(lr) + "_y" + str(y) + ".txt"
log = open(filename,'w')

# Number of runs
num_runs = 2000
# Max number of steps
maxSteps = 100

# What our game world looks like
# -1 Loss
# 1 Win
# 0 Open Space
gameMap = [-1,0,0,0,1,0,-1]

# Number of possible states you can be in our world
# since the only thing that mattters is the position on the map
# we end up with len(gameMap) states
stateCount = len(gameMap)

# number of actions in our world
# AKA Left or Right = 2
actionCount = 2

# Clears the default graph stack and resets the global default graph.
tf.reset_default_graph()

# Create the input array
inputs1 = tf.placeholder(shape=[1,stateCount], dtype=tf.float32)

# Weights I think, in range [0,0.01)
# Not sure why the example uses random_uniform instead of any of the other random options

W = tf.Variable(tf.random_uniform([stateCount, actionCount], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout,1)

nextQ = tf.placeholder(shape=[1,actionCount], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ-Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

jList = []
rList = []
e = 0.1


with tf.Session() as sess:
  sess.run(init)
  for i in range(num_runs):

    # pick a random starting position
    s = np.random.randint(0, stateCount)
    #if that is not a valid starting position retry until we obtain a valid position
    while(gameMap[s] != 0):
      s = np.random.randint(0, stateCount)

    rAll = 0
    j = 0

    if debug==1:
      log.write("Round " + str(i)+"\n")
      log.write("----------\n")


    while j < maxSteps:
      j += 1

      # Definitely not sure what this is doing. a is action I think
      a, allQ = sess.run([predict,Qout], feed_dict={inputs1:np.identity(stateCount)[s:s+1]})

      random = "\n"
      # Small chance that we move randomly
      if np.random.rand(1)<e:
        random = "      RANDOM\n"
        if np.random.rand(1)<0.5:
          a[0]==0
        else:
          a[0]==1

      # Get new state and reward from action
      if a[0] == 0:
        s1 = s-1
      else:
        s1 = s+1
      r = gameMap[s1]

      if debug==1:
        check = sess.run(W)
        log.write(np.array_str(check[s,:]))
        log.write(" Current state: " + str(s) + "   Moving to: " + str(s1) + "   r: " + str(r) + random)

      # Get the Q' values by running the new state through the network
      Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(stateCount)[s1:s1+1]})

      # Obtain maxQ' and set the target value for chosen action
      maxQ1 = np.max(Q1)
      targetQ = allQ
      targetQ[0,a[0]] = r + y*maxQ1

      # Train the network
      _, W1 = sess.run([updateModel, W], feed_dict={inputs1:np.identity(stateCount)[s:s+1],nextQ:targetQ})

      rAll += r
      s = s1
      if r==1 or r==-1:
        #reduce chance of random action as we learn
        e = 1./((i/50)+10)
        break 
    if debug==1:
      log.write("\n")
      #input()
    jList.append(j)
    rList.append(rAll)      

log.close()
print("Score over time: " + str(sum(rList)/num_runs))
