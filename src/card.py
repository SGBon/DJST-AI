# classes and functions for cards and decks of cards

import random

cardNumbers = ["ace",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "jack",
    "queen",
    "king"]

cardSuits = ["spades","hearts","diamonds","clubs"]

blackjackValues = [1,2,3,4,5,6,7,8,9,10,10,10,10]

# represents a card in the blackjack game
class Card:
    def __init__(self,number,suit,value):
        self.number = number
        self.suit = suit
        self.value = value

    def __str__(self):
        return "<%s:%s:%d>" % (self.number,self.suit,self.value)

# a deck of cards that can be drawn from and shuffled
class Deck:
    def __init__(self):
        self.reset()

    def __str__(self):
        ret = "["
        for card in self.cards:
            ret += " %s " % card
        ret += "]"
        return ret

    # reset the deck of cards
    def reset(self):
        self.cards = []
        for suit in cardSuits:
            for i in range(len(cardNumbers)):
                self.cards.append(Card(cardNumbers[i],suit,blackjackValues[i]))

    # shuffle the deck
    def shuffle(self):
        random.shuffle(self.cards)

    # remove a card from top of deck and return it
    def draw(self):
        return self.cards.pop()

# test
if __name__ == "__main__":
    deck = Deck()
    print (deck)
    print ("DRAWING")
    print (deck.draw())
    print ("DECK AFTER DRAW")
    print (deck)
