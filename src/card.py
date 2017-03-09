# classes and functions for cards and decks of cards

import random

cardValues = ["ace",
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

# represents a card in the blackjack game
class Card:
    def __init__(self,value,suit):
        self.value = value
        self.suit = suit

    def __str__(self):
        return "<%s:%s>" % (self.value,self.suit)

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
            for value in cardValues:
                self.cards.append(Card(value,suit))

    # shuffle the deck
    def shuffle(self):
        random.shuffle(self.cards)

    # remove a card from top of deck and return it
    def draw(self):
        return self.cards.pop()

# test
if __name__ == "__main__":
    deck = Deck()
    print deck
    print "DRAWING"
    print deck.draw()
    print "DECK AFTER DRAW"
    print deck
