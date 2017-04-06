# black jack player that chooses actions randomly
# used as basis to compare against neural network

import card
import random
import argparse

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
    push = 0
    deck = card.Deck()
    for i in range(num_runs):
        # reset environment
        deck.reset()
        deck.shuffle()

        dealer = []

        # Deal cards to dealer and player
        hand = []
        hand.append(deck.draw())
        dealer.append(deck.draw())
        hand.append(deck.draw())
        dealer.append(deck.draw())

        #print("Starting hand %s %s" %(hand[0],hand[1]))
        j = 0
        dealer_val = -1
        passed = False
        while j < max_steps:
            j += 1

            val, soft = handValue(hand)

            if val == 21:
                # You got blackjack
                r = 1
                passed = True

            if(not passed):
                # roll for random movement
                a = random.choice(range(actionCount))

                # get new state and reward
                if a == 0: # HIT
                    card = deck.draw()
                    hand.append(card)
                elif a == 1: # PASS
                    passed = True

                val, _ = handValue(hand)

                if val > 21:
                    r = -1
                    passed = False
                else:
                    r = 0


            # perform dealer logic
            if passed:
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
                    r = 1
                elif (dealer_val > val):
                    r = -1
                else:
                    r = 0.5


            if r != 0:
                if r > 0.5:
                    wins += 1
                elif r == 0.5:
                    push += 1
                break

    print("Pushes: "+str(push))
    print("Win percentage\n%f" %(wins/float(num_runs)))
