# black jack player that chooses actions randomly
# used as basis to compare against neural network

import card
import random
import argparse

def handValue(hand):
    val = 0
    for i in hand:
        val += i.value
    return val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BlackJack Neural Network Tester')
    parser.add_argument('num_runs',type=int,help="number of runs to test the model")
    args = parser.parse_args()

    # run parameters
    num_runs = args.num_runs
    max_steps = 10

    # actions in world: hit, stay; split comes later
    actionCount = 2

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

        #print("Starting hand %s %s" %(hand[0],hand[1]))
        j = 0
        while j < max_steps:
            j += 1

            # roll for random movement
            a = random.choice(range(actionCount))

            passed = False
            # get new state and reward
            if a == 0: # HIT
                card = deck.draw()
                hand.append(card)
            elif a == 1: # PASS
                passed = True

            if handValue(hand) == 21:
                # larger reward for getting 21 before dealer
                r = 2
            elif handValue(hand) > 21:
                # larger penalty for busting
                r = -2
            else:
                r = 0


            # perform dealer logic
            if passed:
                dealer.append(deck.draw())
                dealer.append(deck.draw())
                while handValue(dealer) < handValue(hand):
                    dealer.append(deck.draw())
                if handValue(dealer) > 21:
                    # dealer bust, player wins
                    r = 1
                else:
                    r = -1 # dealer beat player

            if r != 0:
                if r > 0:
                    wins += 1
                break

    print("Win percentage %f" %(wins/float(num_runs)))
