import random

class RPSGame:
    """
    A class to encapsulate the game logic for Rock, Paper, Scissors (RPS).
    It includes strategies to play against different opponents with predetermined behaviors.
    """
    
    INITIAL_PLAY = 'S'
    IDEAL_RESPONSE = {'P': 'R', 'R': 'S', 'S': 'P'}
    OPPONENT_BEHAVIORS = {
        'quincy': ['R', 'P', 'P', 'S', 'R'],
        'abbey': ['P', 'P', 'R', 'R', 'R'],
        'kris': ['P', 'R', 'R', 'R', 'R'],
        'mrugesh': ['R', 'R', 'R', 'R', 'R'],
    }
    PLAY_ORDER_TEMPLATE = {
        combo: 0 for combo in ['RR', 'RP', 'RS', 'PR', 'PP', 'PS', 'SR', 'SP', 'SS']
    }

    def __init__(self):
        """Initializes the game state."""
        self.reset_state()

    def reset_state(self):
        """Resets the game state to start a new game or series of games."""
        self.my_history = []
        self.opponent_history = []
        self.prev_play = self.INITIAL_PLAY
        self.opponent_list = [False] * 4
        self.play_order = [dict(self.PLAY_ORDER_TEMPLATE)]
        self.opponent_quincy_counter = -1

    def identify_opponent(self):
        """
        Identifies the opponent's strategy by comparing the last five plays
        against known opponent behavior patterns.
        """
        for i, behavior in enumerate(self.OPPONENT_BEHAVIORS.values()):
            if self.opponent_history[-5:] == behavior:
                self.opponent_list[i] = True
                break

    def quincy_strategy(self):
        """
        Strategy for playing against Quincy.
        Cycles through a predetermined sequence of plays.
        """
        opponent_quincy_list = ['P', 'S', 'S', 'R', 'P']
        self.opponent_quincy_counter = (self.opponent_quincy_counter + 1) % 5
        return opponent_quincy_list[self.opponent_quincy_counter]

    def abbey_strategy(self):
        """
        Strategy for playing against Abbey.
        Analyzes the last two plays to predict Abbey's next move and chooses the optimal counter-play.
        """
        if len(self.my_history) < 2:
            # For the first couple of moves, it might be beneficial to start with a pattern
            # that doesn't give away easy predictions. Let's start with 'P' assuming Abbey might start with 'R'.
            return 'P'
        else:
            last_two = ''.join(self.my_history[-2:])
            if len(last_two) == 2:
                self.play_order[0][last_two] += 1
            
            # Analyze the pattern and predict Abbey's next move
            potential_plays = [self.prev_play + 'R', self.prev_play + 'P', self.prev_play + 'S']
            sub_order = {k: self.play_order[0][k] for k in potential_plays if k in self.play_order[0]}
            if sub_order:
                prediction = max(sub_order, key=sub_order.get)[-1:]
            else:
                # If no clear pattern is established, fall back to a basic counter to the most recent play
                prediction = self.IDEAL_RESPONSE[self.opponent_history[-1]]
            
            # To disrupt Abbey's predictions, occasionally throw a move that counters the counter to your last move
            if len(self.my_history) % 5 == 0:
                # Every fifth move, counter the counter to the last move to introduce unpredictability
                predicted_counter_to_my_last_move = self.IDEAL_RESPONSE[self.my_history[-1]]
                move = self.IDEAL_RESPONSE[predicted_counter_to_my_last_move]
            else:
                move = self.IDEAL_RESPONSE[prediction]

            return move


    def kris_strategy(self):
        """
        Strategy for playing against Kris.
        Simply plays the ideal counter to the previous play.
        """
        return self.IDEAL_RESPONSE[self.prev_play]

    def mrugesh_strategy(self):
        """
        Strategy for playing against Mrugesh.
        Analyzes the most frequent play in the last ten moves and counters it.
        """
        last_ten = self.my_history[-10:]
        most_frequent = max(set(last_ten), key=last_ten.count)
        return self.IDEAL_RESPONSE[most_frequent]

    def player(self, prev_opponent_play):
        """
        The main player function that decides which strategy to use based on the opponent's play history.
        
        :param prev_opponent_play: The last play made by the opponent.
        :return: The play chosen by this function.
        """
        self.opponent_history.append(prev_opponent_play)
        self.my_history.append(self.prev_play)

        if len(set(self.opponent_list)) == 1:
            self.identify_opponent()

        if self.opponent_list[0]:
            self.prev_play = self.quincy_strategy()
        elif self.opponent_list[1]:
            self.prev_play = self.abbey_strategy()
        elif self.opponent_list[2]:
            self.prev_play = self.kris_strategy()
        elif self.opponent_list[3]:
            self.prev_play = self.mrugesh_strategy()
        else:
            self.prev_play = self.INITIAL_PLAY

        if len(self.opponent_history) % 1000 == 0:
            self.reset_state()

        return self.prev_play

game_instance = RPSGame()

def player(prev_opponent_play):
    """
    Wrapper function for the RPSGame class's player method, enabling integration with the game engine.
    
    :param prev_opponent_play: The last play made by the opponent.
    :return: The play chosen by the RPSGame instance.
    """
    return game_instance.player(prev_opponent_play)
