import numpy as np
import tensorflow as tf

# To avoid printing in scientific notation
np.set_printoptions(suppress=True)

# Set up a debug mode
debug = 0

# Q Table
Q = np.zeros([7,2])
# Learning Rate
lr = .9
# Future Reward value
y = .9
# Number of runs
num_runs = 2000
# Max number of steps
maxSteps = 100

# What our game world looks like
# -1 Loss
# 1 Win
# 0 Open Space
gameMap = [-1,0,0,0,1,0,-1]

# Clears the default graph stack and resets the global default graph.
tf.reset_default_graph()

# Create the input array
inputs1 = tf.placeholder(shape=[1,7], dtype=tf.float32)

# Weights I think, in range [0,0.01)
# Not sure why the example uses random_uniform instead of any of the other random options

W = tf.Variable(tf.random_uniform([7,2], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout,1)

nextQ = tf.placeholder(shape=[1,2], dtype=tf.float32)
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
    s = np.random.choice([1,2,3,5])
    rAll = 0
    j = 0

    if debug==1:
      print("Round " + str(i))
      print("----------")

    while j < maxSteps:
      j += 1

      # Definitely not sure what this is doing. a is action I think
      a, allQ = sess.run([predict,Qout], feed_dict={inputs1:np.identity(7)[s:s+1]})

      random = ""
      # Small chance that we move randomly
      if np.random.rand(1)<e:
        random = "      RANDOM"
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
        print(check[s,:])
        print("Current state: " + str(s) + "   Moving to: " + str(s1) + "   reward will be: " + str(r) + random)

      # Get the Q' values by running the new state through the network
      Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(7)[s1:s1+1]})

      # Obtain maxQ' and set the target value for chosen action
      maxQ1 = np.max(Q1)
      targetQ = allQ
      targetQ[0,a[0]] = r + y*maxQ1

      # Train the network
      _, W1 = sess.run([updateModel, W], feed_dict={inputs1:np.identity(7)[s:s+1],nextQ:targetQ})

      rAll += r
      s = s1
      if r==1 or r==-1:
        #reduce chance of random action as we learn
        e = 1./((i/50)+10)
        break
    if debug==1:
      print()
      input()
    jList.append(j)
    rList.append(rAll)      

print("Score over time: " + str(sum(rList)/num_runs))
