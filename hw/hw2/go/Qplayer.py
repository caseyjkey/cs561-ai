import random
import sys
import numpy as np
from read import readInput
from write import writeOutput

from host import GO

Piece = { "BLACK": 1, "WHITE": 2 }
Reward = { "DRAW": 0, "WIN": 100, "LOSS": -100 }

class QPlayer():
    GAME_NUM = 100000

    def __init__(self, alpha=.7, gamma=.9, initial_value=0.5, piece_type=None):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.type = 'Q'
        # 1('X') or 2('O')
        self.piece_type = piece_type
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self.history_states = []
        self.initial_value = initial_value
        # self.state = ?

    def Q(self, state):
        if state not in self.q_values:
            q_val = np.zeros((5, 5))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]
    
    def _select_best_move(self, board):
        print(board)
        print(type(board))
        state = [cell for row in board for cell in row]
        state = frozenset(state)
        q_values = self.Q(state)
        row, col = 0, 0
        curr_max = -np.inf
        while True:
            i, j = self._find_max(q_values)
            print(i, j)
            if go.valid_place_check(i, j, self.piece_type, test_check = True):
                return i, j
            else:
                q_values[i][j] = -1.0

    def _find_max(self, q_values):
        curr_max = -np.inf
        row, col = 0, 0
        for i in range(0, 5):
            for j in range (0, 5):
                if q_values[i][j] > curr_max:
                    curr_max = q_values[i][j]
                    row, col = i, j
        return row, col

    def get_input(self, go):
        '''
        Get one input.

        :param go: Go instance.
        :return: (row, column) coordinate of input.
        '''
        board = go.board
        row, col = self._select_best_move(board)
        self.history_states.append((board, (row, col)))

        return (row, col)
    
    # called when game ends
    def learn(self, board, result):
        if result == self.piece_type:
            result = Reward.WIN
        elif result == 0:
            result = Reward.DRAW
        else:
            result = Reward.LOSS
        self.history_states.reverse()
        max_q_value = -1.0
        for hist in self.history_states:
            state, move = hist
            q = self.Q(state)
            if max_q_value < 0:
                q[move[0]][move[1]] = reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            max_q_value = np.max(q)
        self.history_states = []

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = QPlayer(piece_type=piece_type)
    action = player.get_input(go)
    if not go.place_chess(action[0], action[1], piece_type):
        # Invalid move is immediate loss
        result = Piece.BLACK if piece_type == Piece.WHITE else Piece.WHITE
        player.learn(board, result)
    if go.game_end(piece_type, action):       
        result = go.judge_winner()
        player.learn(board, result)
    writeOutput(action)