# runs test on pretrained model to play blackjack

import card
import numpy as np
import random
import tensorflow as tf
import argparse
import nninfo

def handValue(hand):
    val = 0
    soft = 0
    for i in hand:
        if(i.number == "ace"):
            soft += 1
        val += i.value
        if(val > 21 and soft > 0):
            val -= 10
            soft -= 1
    return val, soft

"""
def display(hand, dealer):
    for i in dealer:
        img = cv2.imread('images/'+i.number+'_of_'+i.suit+'.png', 0)
        img = cv2.resize(img, (120, 180))
        if(i is dealer[0]):
            game_img = img
        else:
            game_img = np.concatenate((game_img, img), axis=1)
    for i in hand:
        img = cv2.imread('images/'+i.number+'_of_'+i.suit+'.png', 0)
        img = cv2.resize(img, (120, 180))
        if(i is hand[0]):
            player_img = img
        else:
            player_img = np.concatenate((player_img, img), axis=1)
    if 'game_img' in locals():
        gh, gw = game_img.shape
        ph, pw = player_img.shape
        height = gh + ph
        width = max(gw, pw)
        tmp = np.zeros((height, width), np.uint8)
        tmp[0:gh,0:gw] = game_img
        tmp[gh:height,0:pw] = player_img
        game_img = tmp
        #game_img = np.concatenate((game_img, player_img), axis=0)
        print("Dealer's Hand\nPlayer's Hand")
    else:
        game_img = player_img
        print("Player's Hand\n")
    cv2.imshow('',game_img)
    # press any key to continue
    cv2.waitKey(100)
    cv2.destroyAllWindows()
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BlackJack Neural Network Tester')
    parser.add_argument('num_runs',type=int,help="number of runs to test the model")
    parser.add_argument('modelfile', help='file where trained model is located')
    args = parser.parse_args()

    # To avoid printing in scientific notation
    np.set_printoptions(suppress=True)

    # run parameters
    num_runs = args.num_runs
    max_steps = 10
    # 0 - num_runs, changes which run will be displayed
    display_run = 0

    lr = 0.01 # learning rate
    y = .9 # future reward value

    # possible states, currently hand value and bust state (1-22)
    stateCount = 29

    # actions in world: hit, stay; split comes later
    actionCount = 2

    # clears default graph stack
    tf.reset_default_graph()

    # establish input tensor and weights used for actions
    inputs = tf.placeholder(shape=[1,stateCount],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([stateCount,actionCount],0,0.001))
    Qout = tf.matmul(inputs,W)
    predict = tf.argmax(Qout,1)

    # establish loss functions that our neural network will optimize
    nextQ = tf.placeholder(shape=[1,actionCount],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    updateModel = trainer.minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,args.modelfile)

        wins = 0
        push = 0
        deck = card.Deck()
        for i in range(num_runs):
            # reset environment
            deck.reset()
            deck.shuffle()

            dealer = []

            # starting hand
            hand = []
            hand.append(deck.draw())
            dealer.append(deck.draw())
            hand.append(deck.draw())
            dealer.append(deck.draw())
            #if (i == display_run):
            #    display(hand, dealer)

            # get value of state
            val, soft = handValue(hand)
            s = nninfo.statemap[(val, soft)]
            j = 0

            passed = False
            dealer_val = -1
            while j < max_steps:
                j += 1
                # choose an action greedily with e chance of random action
                a,allQ = sess.run([predict,Qout],feed_dict={inputs:np.identity(stateCount)[s:s+1]})

                if val == 21:
                    # You got blackjack
                    r = 1
                    passed = True

                if(not passed):
                    # get new state and reward
                    if a[0] == 0: # HIT
                        card = deck.draw()
                        hand.append(card)
                       # if (i == display_run)f:
                        #    display(hand, dealer)
                        val, soft = handValue(hand)
                        if(val < 21):
                            s1 = nninfo.statemap[(val, soft)]
                        else:
                            s1 = 28
                    elif a[0] == 1: # PASS
                        s1 = s
                        passed = True

                    if val > 21:
                        # larger penalty for busting
                        r = -2
                        s1 = 28
                        passed = False
                    else:
                        s1 = s
                        r = 0

                # perform dealer logic
                if passed:
                    #if (i == display_run):
                     #   display(hand, dealer)

                    # Determine value of dealers current hand
                    dealer_val, dealer_soft = handValue(dealer)

                    # If dealer has blackjack everyone losses
                    if(dealer_val == 21):
                        r = -1

                    # Hit on soft 17 stand elsewise
                    while ((dealer_val < 17) or (dealer_val == 17 and dealer_soft > 0)):
                        dealer.append(deck.draw())
                        dealer_val, dealer_soft = handValue(dealer)

                    if (dealer_val > 21) or (val > dealer_val):
                        # dealer bust, player wins
                        r = 2
                    elif (dealer_val > val):
                        r = -1
                    else:
                        r = 0.5


                s = s1 # update state
                if r != 0:
                    if r > 0.5:
                        wins += 1
                    elif r == 0.5:
                        push += 1
                    # reduce chance of random action
                    e = 1./((i/50.0)+10)
                    break

        print("Pushes: "+str(push))
        print("Win percentage\n%f" %(wins/float(num_runs)))
