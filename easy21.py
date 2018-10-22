import numpy as np
from tqdm import tqdm

class CardGame():
    """Base class for the card game"""
    def __init__(self):
        # Initialize the cards deck
        self.card_deck = self.initialize_deck()
        # Initialize the game state
        self.initialize_game()

    def reset_game(self):
        self.__init__()

    def initialize_deck():
        pass

    def initialize_game():
        pass

"""Question 1"""
class Easy21(CardGame):
    """Easy 21 Game Class"""

    def initialize_deck(self):
        """
        1 to 10 color red probability 1/3
        1 to 10 color black probability 2/3

        >>> game = Easy21()
        >>> 2 in game.card_deck
        True
        >>> len(game.card_deck) == 30
        True
        >>> np.max(game.card_deck) == 10
        True
        """
        cards = [(card, pattern) 
                for card, pattern 
                in zip(list(range(1,11))*3, list(range(1,4))*10)]
        return np.array(cards)

    def _draw_card(self):
        """
        Returns a numpy array with two elements:
        Random Card Value between 1-10 and Card Type between 1-3
        """
        idx = np.random.randint(0, 30)
        card = self.card_deck[idx]
        return card

    def _play_player(self, player_sum):
        """
        Returns the sum of the player's hand after drawing a new card.
        """
        card = self._draw_card()
        return player_sum + self.get_card_value(card)

    def _play_dealer(self, dealer_sum):
        """
        Returns the sum of the dealers's hand after following his strategy.
        """
        while dealer_sum < 17:
            card = self._draw_card()
            dealer_sum += self.get_card_value(card)
        return dealer_sum

    def initialize_game(self):
        player_card = self._draw_card()
        dealer_card = self._draw_card()
        player_card_value = self.get_card_value(player_card)
        dealer_card_value = self.get_card_value(dealer_card)
        self.current_state = (dealer_card_value, player_card_value)
        self.isTerminal = False
        self.playerBusted = False

    def get_card_value(self, card):
        card_type = self.get_card_type(card)
        if card_type == 0:
            return -card[0]
        return card[0]

    def get_card_type(self, card):
        if card[1] == 1:
            return 0 # red card return negative value
        return 1 # black card return positive value

    def play_until_evaluate_state(self):
        while self.current_state[1] < 12:
            self.current_state, _ = self.step(self.current_state, 'hit')

    def step(self, current_state, action):
        """
        Takes as input a state s and an action a and 
        returns a sample of the next state s' and reward r.
    
        Input:
        current_state:   Tuple -> Dealer's first card 1-10 and the player's 
                                  sum 1-21 
        action:   Action -> hit or stick 
        Output:
        new_state:   Tuple -> Next or terminal state  (Dealer's first card 
                              and the player's sum 1-21)
        reward       reward during this step 
        """
        dealer_sum, player_sum = current_state

        if action == "stick":
            dealer_sum = self._play_dealer(dealer_sum) 
            self.isTerminal = True
        elif action == "hit":
            player_sum = self._play_player(player_sum)
            if player_sum > 21:
                self.isTerminal = True
                self.playerBusted = True

        if self.isTerminal and (self.playerBusted 
                                or player_sum < dealer_sum < 21):
            reward = -1
        elif self.isTerminal and (player_sum > dealer_sum or dealer_sum > 21):
            reward = 1
        else:
            reward = 0

        new_state = (dealer_sum, player_sum)
        return new_state, reward

    def play(self):
        self.initialize_game()
        dealer_sum, player_sum = self.current_state
        print(f"Dealer has {dealer_sum} and you {player_sum}")
        action = input("Give 0 to stick and 1 to hit:\n")
        while action == '1':
            self.current_state, _ = self.step(self.current_state, 'hit')
            dealer_sum, player_sum = self.current_state
            if not self.isTerminal:
                print(f"Dealer has {dealer_sum} and you {player_sum}")
                action = input("Give 0 to stick and 1 to hit:\n")
            else:
                print(f"Dealer has {dealer_sum} and you {player_sum}")
                return 1
        self.current_state, _ = self.step(self.current_state, 'stick')
        print(f"Dealer has {dealer_sum} and you {player_sum}\n")
        return 1




        


class BaseAgent():
    """ Agent base class"""
    def __init__(self, game='Easy21', alpha=1.0, epsilon=1.0, N0 = 100,
                gamma = 1.0):
        if game == 'Easy21':
            self.game = Easy21()
            state_value_space = (20, 10)
            action_value_space = (20, 10, 2)
            self.get_dealer_index = lambda s: s[0] + 10 if s[0] < 0 else s[0] + 9
            self.get_player_index = lambda s: s[1] - 12
        else:
            raise Exception("You need to define a valid game.")

        self.state_value = np.zeros(state_value_space) 
        self.action_value = np.zeros(action_value_space) 
        self.alpha = np.zeros(action_value_space) + alpha
        self.epsilon = np.zeros(state_value_space) + epsilon
        self.N0 = N0 # N0=100 as per proposed exploration strategy
        self.action_tree = []
        self.rewards_tree = []
        self.gamma = gamma

    def get_policy(self):
        state = self.get_state_idx()
        return self.improve_greedy(state)

    def get_state_idx(self):
        dealer_idx = self.get_dealer_index(self.game.current_state)
        player_idx = self.get_player_index(self.game.current_state)
        return dealer_idx, player_idx

    def improve_greedy(self, state):
        return np.argmax(self.action_value[state])

    def improve_egreedy(self, state):
        if self.epsilon[state] < np.random.random():
            return self.improve_greedy(state)
        action_space = np.arange(len(self.action_value[state]))
        return np.random.choice(action_space)

    def improve_policy(self, state, method='greedy'):
        if method == 'egreedy':
            return self.improve_egreedy(state)
        elif method == 'greedy':
            return self.improve_greedy(state)
        else:
            raise Exception("You need to define a valid improvement method")

    def evaluate_policy(self):
        """Method defined localy to each agent subclass"""
        pass


    def play(self):
        """Method defined localy to each agent subclass"""
        pass

    def update_game_counter(self, goal):
        if goal == 1:
            self.win_count += 1
        elif goal == -1:
            self.lose_count += 1

    def reset_game_counter(self):
        self.win_count = 0
        self.lose_count = 0

    def reset_action_tree(self):
        self.action_tree = []

    def update_action_tree(self, value):
        self.action_tree.append(value)

    def reverse_action_tree(self):
        self.action_tree.reverse()

    def update_rewards_tree(self, reward):
        self.rewards_tree.append(reward)

    def reset_rewards_tree(self):
        self.rewards_tree = []

    def reverse_rewards_tree(self):
        self.rewards_tree.reverse()
        
    def get_action(self, action_idx):
        if action_idx == 0:
            return 'stick'
        elif action_idx == 1:
            return 'hit'

"""Question 2"""
class MonteCarloAgent(BaseAgent):

    def evaluate_and_improve_policy(self):
        self.game.play_until_evaluate_state()
        while not self.game.isTerminal: 
            dealer_idx, player_idx = self.get_state_idx()
            action_idx = self.improve_policy((dealer_idx, player_idx), 
                                              method='egreedy')
            state_idx = (dealer_idx, player_idx, action_idx)
            self.update_action_tree(state_idx)
            action = self.get_action(action_idx)

            self.game.current_state, reward = self.game.step(
                                                    self.game.current_state, 
                                                    action)

            self.update_rewards_tree(reward)
            self.game.play_until_evaluate_state()


        self.evaluated_state = []
        goal = 0
        for tree, reward in zip(self.action_tree, self.rewards_tree):
            self.epsilon[tree[0], tree[1]] = self.N0 \
                    / (self.N0 + np.sum(self.alpha[tree[0], tree[1]]))

            goal += self.gamma * reward
            if tree not in self.evaluated_state: # one step Monte Carlo
                self.action_value[tree] = self.action_value[tree] \
                                        + 1/self.alpha[tree] \
                                        * (goal - self.action_value[tree])
                self.alpha[tree] += 1
                self.evaluated_state.append(tree)

    def monte_carlo_control(self):
        for _ in tqdm(range(1000000)):
            self.game.initialize_game()
            self.evaluate_and_improve_policy()

    def monte_carlo_play(self):
        self.epsilon[:] = 0
        self.win_count = 0
        self.lose_count = 0
        for _ in range(100000):
            self.game.initialize_game()
            self.evaluate_policy()

"""Question 3"""
