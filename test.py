import sys
from collections import deque
import random

# Mock the imports or add paths so we can import from game
sys.path.append("/Users/tvg/Desktop/01-Projects/04-Game-Dev/03-Clash-Royale")

from game.player_side import PlayerSide

def test_deck_cycling():
    player = PlayerSide()
    print("Initial hand:", [c.__name__ for c in player.hand])
    print("Initial next card:", player.next_card.__name__)
    print("Queue:", [c.__name__ for c in player.card_queue])
    
    # Let's deploy the first card in hand
    active_idx = 0
    used_card = player.hand[active_idx]
    next_card = player.next_card
    
    print(f"\nDeploying card: {used_card.__name__} from slot {active_idx}")
    player.use_card(active_idx)
    
    print("New hand:", [c.__name__ for c in player.hand])
    print("New next card:", player.next_card.__name__)
    print("Queue:", [c.__name__ for c in player.card_queue])
    
    # Assertions to verify correctness
    assert player.hand[active_idx] == next_card, "The next card should now be in the hand where the used card was"
    assert player.card_queue[-1] == used_card, "The used card should be at the back of the queue"
    assert player.next_card == player.card_queue[0], "next_card should match the front of the queue"
    
    print("\nTest passed successfully!")

if __name__ == "__main__":
    test_deck_cycling()