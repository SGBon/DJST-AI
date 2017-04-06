# runs test on pretrained model to play blackjack

import card
import numpy as np
import random
import tensorflow as tf
import numpy as np
import cv2
import argparse

def handValue(hand):
    val = 0
    for i in hand:
        val += i.value
    return val

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
    stateCount = 22

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
        deck = card.Deck()
        for i in range(num_runs):
            # reset environment
            deck.reset()
            deck.shuffle()

            dealer = []

            # starting hand
            hand = []
            hand.append(deck.draw())
            hand.append(deck.draw())
            if (i == display_run):
                display(hand, dealer)

            # get value of state
            s = handValue(hand) - 1
            j = 0

            #print("Starting hand %s %s" %(hand[0],hand[1]))
            while j < max_steps:
                j += 1
                # choose an action greedily with e chance of random action
                a,allQ = sess.run([predict,Qout],feed_dict={inputs:np.identity(stateCount)[s:s+1]})

                passed = False
                # get new state and reward
                if a[0] == 0: # HIT
                    card = deck.draw()
                    hand.append(card)
                    if (i == display_run):
                        display(hand, dealer)
                    s1 = handValue(hand)
                elif a[0] == 1: # PASS
                    s1 = s
                    passed = True

                if s1 == 21:
                    # larger reward for getting 21 before dealer
                    r = 2
                elif s1 > 21:
                    # larger penalty for busting
                    r = -2
                    s1 = 22
                else:
                    r = 0

                s1 -= 1

                # perform dealer logic
                if passed:
                    dealer.append(deck.draw())
                    dealer.append(deck.draw())
                    if (i == display_run):
                        display(hand, dealer)
                    while handValue(dealer) < handValue(hand):
                        dealer.append(deck.draw())
                        if (i == display_run):
                            display(hand, dealer)
                    if handValue(dealer) > 21:
                        # dealer bust, player wins
                        r = 1
                    else:
                        r = -1 # dealer beat player

                s = s1 # update state
                if r != 0:
                    if r > 0:
                        wins += 1
                    break

        print("Win percentage\n%f" %(wins/float(num_runs)))
