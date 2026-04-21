import torch
from torch import nn
import random

class Deck:
    def __init__(self,device):
        self.device = device
        self.decks = 8
        self.running_count = 0  # Was calculating every time with a function a variable helps w runtime
        self.card_counts = {}  # Dictionary also saves runtime had a function that computed from a list iteration with was not efficient
        self.hand = []
        self.dealer = []
        self.stake = 0
        self._fresh_shoe()

    ###########################
    # State functions, functions to help build state tensor
    ###########################

    def add_card_player(self,card):
        card = int(card) if card != "A" else card
        self.hand.append(card)
        self.card_counts[card] -= 1
        self._reveal(card)

    def add_card_dealer(self,card):
        card = int(card) if card != "A" else card
        self.dealer.append(card)
        self.card_counts[card] -= 1
        self._reveal(card)

    def add_stake(self, s):
        self.stake = s

    def clear_hand(self):
        self.hand = []
        self.dealer = []
        self.stake = 0

    def _reveal(self, card):
        if card in [2, 3, 4, 5, 6]:
            self.running_count += 1
        elif card in [10, 'A']:
            self.running_count -= 1

    def get_score_and_soft(self, hand):
        total, aces = 0, 0
        for card in hand:
            if card == 'A':
                aces += 1
                total += 11
            else:
                total += card
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total, int(aces > 0)

    def get_running_count(self):
        return self.running_count

    def get_true_count(self):
        decks_remaining = len(self.cards) / 52
        divisor = max(decks_remaining, 1 / 52)
        return self.running_count / divisor

    def get_hidden_prob(self):
        total = len(self.cards)
        if total == 0:
            return [0] * 10

        probs = [self.card_counts[i] / total for i in range(2, 11)]
        probs.append(self.card_counts['A'] / total)
        return probs

    def get_state_action(self):
        """
        Creates the tensor for the action models input layer
        """

        player_total, is_soft = self.get_score_and_soft(self.hand)

        player_total_norm = player_total / 30  # Normalize over max total
        dealer_up = self._card_value(self.dealer[0]) / 11 if self.dealer else 0  # Normalize over max dealer up
        true_count_norm = self.get_true_count() / 8.0  # Got to 0.8 through trail and error, soft form of normalization
        hidden_prob = self.get_hidden_prob()  # Normalized list of prob of each value being the hidden card based on cards seen so far in deck

        return torch.tensor(
            hidden_prob + [self.stake / 10, player_total_norm, float(is_soft), dealer_up, true_count_norm],
            dtype=torch.float, device=self.device).unsqueeze(0)

    def get_state_bet(self):
        """
        Creates the tensor for the bet models input layer
        """

        true_count_norm = self.get_true_count() / 8.0  # Got to 8 through trail and error, soft form of normalization
        hidden_prob = self.get_hidden_prob()

        return torch.tensor(hidden_prob + [true_count_norm], dtype=torch.float, device=self.device).unsqueeze(0)


    def _card_value(self, card):
        return 11 if card == 'A' else card

    def _fresh_shoe(self):
        shoe = [10] * 16 * self.decks + [9] * 4 * self.decks + [8] * 4 * self.decks + \
               [7] * 4 * self.decks + [6] * 4 * self.decks + [5] * 4 * self.decks + \
               [4] * 4 * self.decks + [3] * 4 * self.decks + [2] * 4 * self.decks + ['A'] * 4 * self.decks
        random.shuffle(shoe)
        self.cards = shoe

        self.running_count = 0
        self.count = []
        self.card_counts = {i: 4 * self.decks for i in range(2, 10)}
        self.card_counts[10] = 16 * self.decks
        self.card_counts['A'] = 4 * self.decks
        return shoe

class NeuralNetwork(nn.Module):
    """
    Creates the individual MLPs
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.out(x)

device = torch.device("mps" if torch.mps.is_available() else "cpu")

bet_model = NeuralNetwork(11, 4)
action_model = NeuralNetwork(15, 4)

bet_model.load_state_dict(torch.load("Models/bet.pth", weights_only=True, map_location=torch.device(device)))
action_model.load_state_dict(torch.load("Models/action.pth", weights_only=True, map_location=torch.device(device)))

bet_model.to(device)
action_model.to(device)
bet_model.eval()
action_model.eval()

STOP_FLAG = False

deck = Deck(device)

def handler():
    handle = int(input("""
    0: Add Card to Player Hand or Count At the End
    1: Add Card to Dealer Hand 
    2: Add Hand Stake
    3: Get Bet Prediction 
    4: Get Action Prediction 
    5: Clear Hand
    6: End Game
    """))
    if handle == 0:
        card = input('Enter card A or 2-10: ')
        deck.add_card_player(card)
    elif handle == 1:
        card = input('Enter card A or 2-10: ')
        deck.add_card_dealer(card)
    elif handle == 2:
        stake = float(input('Enter stake 0.1, 0.5, 1, 10: '))
        deck.add_stake(stake)
    elif handle == 3:
        print(f"Perform bet {bet_model(deck.get_state_bet()).argmax(dim=-1).item()}")
    elif handle == 4:
        print(f"Perform action {action_model(deck.get_state_action()).argmax(dim=-1).item()}")
    elif handle == 5:
        deck.clear_hand()
    else:
        global STOP_FLAG
        STOP_FLAG = True

    print(f"Player {deck.hand}")
    print(f"Dealer {deck.dealer}")
    print(f"Stake {deck.stake}")

if __name__ == '__main__':
    """
    Rules:
    1. Deck hand must be cleared to get bet
    2. Hand, Dealer, and Stake must be filled in to get action 
    3. Add the Dealers hidden card to the count at the end
    """

    #Actions Dict
    #0: Hit
    #1: Double
    #2: Stand
    #3: Split

    # Bet Dic0t
    # 0: 0.1
    # 1: 0.5
    # 2: 1.0
    # 3: 10

    while not STOP_FLAG:
        handler()

    print("Game Finished")