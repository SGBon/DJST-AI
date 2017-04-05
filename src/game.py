import card
import numpy as np
import random
import tensorflow as tf

def handValue(hand):
    val = 0
    for i in hand:
        val += i.value
    return val

if __name__ == "__main__":
    num_runs = 10000
    max_steps = 100

    lr = 0.01 # learning rate
    y = .9 # future reward value

    # possible states, currently hand value and bust state (1-22)
    stateCount = 22

    # actions in world: hit, stay; split comes later
    actionCount = 2

    # clears default graph stack
    tf.reset_default_graph()

    # establish input tensor and weights used for actions
    inputs = tf.placeholder(shape=[1,stateCount],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([stateCount,actionCount],0,0.01))
    Qout = tf.matmul(inputs,W)
    predict = tf.argmax(Qout,1)

    # establish loss functions that our neural network will optimize
    nextQ = tf.placeholder(shape=[1,actionCount],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    updateModel = trainer.minimize(loss)

    init = tf.global_variables_initializer()

    # reward and steps list
    jlist = []
    rlist = []
    e = 0.1

    with tf.Session() as sess:
        sess.run(init)
        wins = 0
        deck = card.Deck()
        for i in range(num_runs):
            # reset environment
            deck.reset()
            deck.shuffle()

            # starting hand
            hand = []
            hand.append(deck.draw())
            hand.append(deck.draw())

            # get value of state
            s = handValue(hand) - 1
            rAll = 0
            j = 0

            #print("Starting hand %s %s" %(hand[0],hand[1]))
            while j < max_steps:
                j += 1
                # choose an action greedily with e chance of random action
                a,allQ = sess.run([predict,Qout],feed_dict={inputs:np.identity(stateCount)[s:s+1]})

                # roll for random movement
                if np.random.rand(1) < e:
                    a[0] = random.choice(range(actionCount))

                passed = False
                # get new state and reward
                if a[0] == 0: # HIT
                    card = deck.draw()
                    hand.append(card)
                    s1 = handValue(hand)
                elif a[0] == 1: # PASS
                    s1 = s
                    passed = True

                if s1 == 21:
                    r = 1
                elif s1 > 21:
                    r = -1
                    s1 = 22
                else:
                    r = 0

                s1 -= 1

                # perform dealer logic
                if passed:
                    dealer = []
                    dealer.append(deck.draw())
                    dealer.append(deck.draw())
                    while handValue(dealer) < handValue(hand):
                        dealer.append(deck.draw())
                    if handValue(dealer) > 21:
                        r = 1 # dealer busted, player wins
                    else:
                        r = -1 # dealer beat player

                # obtain Q' values by feeding new state through network
                Q1 = sess.run(Qout,feed_dict={inputs:np.identity(stateCount)[s1:s1+1]})

                # Obtain maxQ' and set target value for chosen action
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r * y*maxQ1

                # train network with target and predicted value
                _, W1 = sess.run([updateModel,W],feed_dict={inputs:np.identity(stateCount)[s:s+1],nextQ:targetQ})

                rAll += r

                s = s1
                if r != 0 or passed == True:
                    # reduce chance of random action
                    e = 1./((i/50.0)+10)
                    #print("Finishing with reward %d and handvalue %d" % (r,s))
                    break
            jlist.append(j)
            rlist.append(rAll)

    print("Average score %f" % (sum(rlist)/float(num_runs)))
