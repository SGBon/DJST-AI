import card

def handValue(hand):
    val = 0
    for i in hand:
        val += i.value
    return val

if __name__ == "__main__":
    deck = card.Deck()
    deck.shuffle()

    # starting hand
    hand = []
    hand.append(deck.draw())
    hand.append(deck.draw())

    print("Starting hand %s %s" %(hand[0],hand[1]))

    while handValue(hand) < 21:
        card = deck.draw()
        print("Hit %s" % card)
        hand.append(card)
    if handValue(hand) == 21:
        print("You win somehow")
    else:
        print("Bust")
