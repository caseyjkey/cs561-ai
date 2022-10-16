import random
import sys
from read import readInput
from write import writeOutput
import cProfile

from host import GO

from copy import deepcopy
import numpy as np
from random import sample, uniform

class AlphaBetaPlayer():
    def __init__(self, pieceType, maxDepth=20, maxActions=np.inf):
        self.type = 'alpha beta'
        # Depth before using heuristic function
        self.maxDepth = maxDepth
        # Limit the possible actions for each node
        self.maxActions = maxActions
        self.turns = 0
        self.pieceType = pieceType
        self.enemy = 1 if pieceType == 2 else 2

    def get_input(self, go):
        '''
        Get one input.

        :param go: Go instance.
        :param pieceType: 1('X')-Black or 2('O')-White.
        :return: (row, column) coordinate of input.
        '''      
        print(f"we are {self.pieceType}")
        depth = self.maxDepth
        maxActions = self.maxActions
        action = sample(self.possibleActions(go), 1)
        goCopy = go.copy_board()
        # Why are we returning a negative value, e.g. -16?
        value, actions = self.maxNode(goCopy, action, 0, -np.inf, np.inf)
        print("val", value, "acts", actions)

        if not actions:
            return "PASS"
        else:
            return actions[0]

    def findGroups(self, go):
        # find endangered groups
        # TODO: each group a dict with liberty count and liberty points
        # {piece: groups [ (numLiberties, set(libertyPoints))]}
        groups = {1: [], 2: []}
        board = go.board
        visited = set()
        for i in range(len(board)):
            for j in range(len(board)):
                if (i, j) not in visited and board[i][j]:
                    color = board[i][j]
                    libertyPoints = set()
                    allies = go.ally_dfs(i, j)
                    for ally in allies:
                        visited.add(ally)
                        points = self.findLiberties(ally[0], ally[1], color, board)
                        libertyPoints = libertyPoints | points
                    groups[color].append((len(libertyPoints), libertyPoints))
        return groups

    # Returns score and possible actions
    # Action should be a coordinate
    def maxNode(self, go, action, depth, alpha, beta):
        actions = self.possibleActions(go)
        if depth >= self.maxDepth or not actions:
            return self.scoreHeuristic(go, actions, not actions), []
            # return self.heuristic(go, actions, not actions), []
        
        maxValue = -np.inf
        maxValueActions = []

        actions = sample(actions, self.maxActions) if len(actions) > self.maxActions else actions

        for action in actions:
            goCopy = go.copy_board()
            valid = goCopy.place_chess(action[0], action[1], self.pieceType)
            score, actions = self.minNode(goCopy, action, depth + 1, alpha, beta)
            maxValue = score if score > maxValue else maxValue
            maxValueActions.insert(0, action)

            if maxValue > beta:
                return maxValue, maxValueActions
            
            alpha = maxValue if maxValue > alpha else alpha
        
        return maxValue, maxValueActions
    
    
    def minNode(self, go, action, depth, alpha, beta):
        actions = self.possibleActions(go)
        if depth >= self.maxDepth or not actions:
            return self.scoreHeuristic(go, action, not actions), []
        
        enemy = 2 if self.pieceType == 1 else 1

        minValue = np.inf
        minValueActions = []

        actions = sample(actions, self.maxActions) if len(actions) > self.maxActions else actions
        for action in actions:
            goCopy = go.copy_board()
            valid = goCopy.place_chess(action[0], action[1], enemy)
            score, actions = self.maxNode(goCopy, action, depth + 1, alpha, beta)
            minValue = score if score < minValue else minValue
            minValueActions.insert(0, action)

            if minValue < alpha:
                return minValue, minValueActions
            
            beta = minValue if minValue < beta else beta
        
        return minValue, minValueActions

    def possibleActions(self, go):
        board = go.board
        actions = []
        for i in range(len(board)):
            for j in range(len(board)):
                if not board[i][j] and go.valid_place_check(i, j, self.pieceType, True):
                    actions.append((i, j))
        return actions

    def scoreHeuristic(self, go, action, finalAction):
        if depth >= self.maxDepth or self.gameOver(go):
            return 


    def gameOver(self, go):
        for i in range(len(go.board)):
            for j in range(len(go.board)):
                if self.openPoint(go, i, j):
                    return False
        return True

    def openPoint(self, go, i, j):
        return not go.board[i][j] and not self.checkPoint(go, i, j)
        
    def score(self, go):
        # :param pieceType: 1('X')-Black or 2('O')-White.
        score = {1: 0, 2: 0}
        for row in range(len(go.board)):
            for col in range(len(go.board)):
                point = self.checkPoint(go, row, col)
                if point:
                    score[point] += 1
        return score
    
    def checkPoint(self, go, i, j):
        point = 0
        if go.board[i][j]:
            point = go.board[i][j]
        else:
            neighbors = go.detect_neighbor(i, j)
            if not any(go.board[x][y] == 0 for (x, y) in neighbors):
                if all(go.board[x][y] == self.pieceType for (x, y) in neighbors):
                    point = self.pieceType
                elif all(go.board[y][y] == self.enemy for (x, y) in neighbors):
                    point = self.enemy
        return point

    def hasNoNeighbors(self, go, i, j):
        neighbors = go.detect_neighbor(i, j)
        return all(go.board[x][y] == 0 for (x, y) in neighbors)
                

    def probabilityHeuristic(self, go, action, finalAction):
        stepCost = 4
        # turns not incrementing correctly
        # TODO: implement using go.board's n_moves property
        value = 20 # - self.turns
        if finalAction:
            winner = go.judge_winner()
            return value if winner == self.pieceType else -1 * value

        groups = self.findGroups(go)
        
        enemy = 2 if self.pieceType == 1 else 1
        
        # good chance of winning
        # if there is at least one enemy group with one or less liberties
        if any(True for group in groups[enemy] if group[0] <= 1):
            # print("good chance winning", action, [group for group in groups[enemy] if group[0] <= 1])
            return value - stepCost
        # good chance of losing
        # if we have more than one group with one or less liberties
        elif sum(1 for group in groups[self.pieceType] if group[0] <= 1) > 1:
            return -1 * (value - stepCost)

        # decent chance of winning
        if self.twoGroupsTwoLibertiesWithOneSharedPoint(enemy, groups):
            return value / 2
        sharedSelfPoints = self.twoGroupsTwoLibertiesWithOneSharedPoint(self.pieceType, groups)
        # if enemy does not share any of these dangerous points, we decent chance of losing
        if sharedSelfPoints and not any(selfPoint in group[1] for group in groups[enemy] for selfPoint in sharedSelfPoints):
            return -1 * value / 2

        # group score
        enemyGroupsWith2Liberties = sum(group[0] == 2 for group in groups[enemy])
        selfGroupsWith2Liberties = sum(group[0] == 2 for group in groups[self.pieceType])
        groupScore = selfGroupsWith2Liberties - enemyGroupsWith2Liberties

        # liberty score
        selfLiberties = set()
        for group in groups[self.pieceType]:
            selfLiberties = selfLiberties | group[1]
        enemyLiberties = set()
        for group in groups[enemy]:
            enemyLiberties = enemyLiberties | group[1]
        libertyScore = len(enemyLiberties) - len(selfLiberties)

        # return a randomly scaled score
        total = groupScore * uniform(0,1) + libertyScore * uniform(0,1)
        return total


        

    def findLiberties(self, i, j, color, board):
        neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        size = len(board)
        count = 0
        liberties = set()
        for neighbor in neighbors:
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                if 0 == board[neighbor[0]][neighbor[1]]:
                    count += 1
                    liberties.add(neighbor)
        
        return liberties
    
    # returns the two groups liberties
    def twoGroupsTwoLibertiesWithOneSharedPoint(self, pieceType, groups):
        # check if a piece has two groups with two liberties and one liberty is shared
        for i, group in enumerate(groups[pieceType]):
            if group[0] == 2:
                firstLiberties = group[1]
                for followingGroup in groups[pieceType][i:]:
                    if followingGroup[0] == 2:
                        secondLiberties = followingGroup[1]
                        if not firstLiberties.isdisjoint(secondLiberties):
                            return firstLiberties | secondLiberties
        return False
        




if __name__ == "__main__":
    N = 5
    pieceType, previous_board, board = readInput(N)
    print('----------consider copying this----------')
    print(board)
    print('-----------------------------------------')
    go = GO(N)
    go.set_board(pieceType, previous_board, board)
    player = AlphaBetaPlayer(pieceType, maxDepth=20, maxActions=5)
    # cProfile.run('action = player.get_input(go)')
    action = player.get_input(go)
    print(action)
    print(go.valid_place_check(action[0], action[1], pieceType))
    writeOutput(action)