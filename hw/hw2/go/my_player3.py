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
        self.startType = pieceType
        self.startEnemy = 1 if pieceType == 2 else 2
        self.pieceType = pieceType
        self.enemy = 1 if pieceType == 2 else 2

    def get_input(self, go):
        '''
        Get one input.

        :param go: Go instance.
        :param pieceType: 1('X')-Black or 2('O')-White.
        :return: (row, column) coordinate of input.
        '''      
        v = self.maxNode(go, 0, [-np.inf], [np.inf])
        a = v[1]
        valid = go.valid_place_check(a[0], a[1], self.startType)
        print(f"we are {self.pieceType}", "acts", v, "valid", valid)
        if not valid:
            sys.exit(1)
            return "PASS"
        else:
            return a

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

    # Returns score and action tuple
    # Action should be a coordinate tuple
    # alpha and beta are subscriptable (become tuple or score and point)
    def maxNode(self, go, depth, alpha, beta):
        if depth >= self.maxDepth or self.gameOver(go):
            score = self.score(go)
            #print("score", score)
            v = score[self.startType] - score[self.startEnemy]
            return [v, None]
        
        #allV = [[None for i in range(len(go.board))] for j in range(len(go.board))]
        v = [-np.inf, None]
        
        alphaCopy = alpha
        betaCopy = beta
        for i in range(len(go.board)):
            for j in range(len(go.board)):
                a = (i, j)
                outcome = self.moveOutcome(go, a, depth, alphaCopy, betaCopy, "min")
                if outcome[0] > v[0]:
                    v = outcome
                    v[1] = a if not v[1] else v[1]
                #allV[i][j] = v
                #print("max")
                #print(go.board)
                #print(v, depth)
                if beta[0] <= v[0]:
                    #print(f'PRUNED BETA BECAUSE {beta[0]} >= {v[0]} ')
                    return v
                alphaCopy = max(alphaCopy, v)

        #allV = [element for row in allV for element in row]        
        return v #max(allV)
    
    def minNode(self, go, depth, alpha, beta):
        if depth >= self.maxDepth or self.gameOver(go):
            score = self.score(go)
            v = score[self.startType] - score[self.startEnemy]
            return [v, None]
        
        #allV = [[None for i in range(len(go.board))] for j in range(len(go.board))]
        v = [np.inf, None]
        alphaCopy = alpha
        betaCopy = beta
        for i in range(len(go.board)):
            for j in range(len(go.board)):
                a = (i, j)
                outcome = self.moveOutcome(go, a, depth, alphaCopy, betaCopy, "max")
                if outcome[0] < v[0]:
                    v = outcome
                    v[1] = a if not v[1] else v[1]
                #allV[i][j] = v
                #print("min")
                #print(go.board)
                #print(v, depth)
                if v[0] <= alphaCopy[0]:
                    #print(f'PRUNED ALPHA BECAUSE {alpha[0]} <= {v[0]} ')
                    return v
                betaCopy = min(betaCopy, v)

        #allV = [element for row in allV for element in row]        
        #print("min")
        #print(allV)
        return v #min(allV)

    # returns a a 2d matrix of outcomes for choosing any point
    # return a point with score if pruning occurs
    def boardOutcomes(self, go, depth, alpha, beta):
        # this is a 2d matrix of whole board
        outcomes = [[None for j in range(len(go.board))] for i in range(len(go.board))]
        alphaCopy = alpha
        betaCopy = beta
        for i in range(len(go.board)):
            for j in range(len(go.board)):
                action = (i, j)
                outcome = self.moveOutcome(go, action, depth, alphaCopy, betaCopy, "")
                #print("inner result for", i, j, "is", outcome, 'depth', depth)
                if outcomes[i][j]:
                    print('ALREADY DEFINED!!!!!!!!!!!!!!!!')
                outcomes[i][j] = outcome
                #print('outcomes', outcomes, '\n')
                #print('alpha, beta', alphaCopy, betaCopy)

                isEven = not bool(depth % 2)
                # These may need switched
                if isEven:
                    if beta[0] >= outcome[0]:
                        print(f'PRUNED BETA BECAUSE {beta[0]} >= {outcome[0]} ')
                        return outcome
                    if alphaCopy[0] > outcome[0]:
                        print('alphaCopy now', outcome)
                        alphaCopy = outcome
                else:
                    if alpha[0] <= outcome[0]:
                        print(f'PRUNED ALPHA BECAUSE {alpha[0]} <= {outcome[0]} ')
                        return outcome
                    if betaCopy[0] < outcome[0]:
                        print('betacopy now', outcome)
                        betaCopy = outcome
        # Flatten so we can extract min/max value
        return [value for row in outcomes for value in row]
                

    # returns a score as (score, (x, y))
    def moveOutcome(self, go, a, depth, alpha, beta, level):
        player = self.pieceType if level == "min" else self.enemy
        if self.isValid(go, a, player):
            goCopy = go.copy_board()
            valid = goCopy.place_chess(a[0], a[1], player)
            #print(a, 'is', valid)
            #print('placing a', self.pieceType, 'at', action, "in depth of", depth)
            if level == "max":
                return self.maxNode(goCopy, depth+1, alpha, beta)
            else:
                return self.minNode(goCopy, depth, alpha, beta)
        else:
            score = -1000 if level == "min" else 1000
            return [score, a]

    def isValid(self, go, action, player):
        (i, j) = action
        valid = True
        if go.board[i][j]:
            valid = False
        elif self.hasNoNeighbors(go, i, j):
            valid = False
        elif self.checkPoint(go, i, j):
            valid = False
        elif not go.valid_place_check(i, j, player):
            valid = False
        print(action, not go.board[i][j], not self.hasNoNeighbors(go, i, j), not self.checkPoint(go, i, j), go.valid_place_check(action[0], action[1], player))
        return valid

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
                player = self.checkPoint(go, row, col)
                if player:
                    score[player] += 1
        return score
    
    def checkPoint(self, go, i, j):
        point = 0
        if go.board[i][j]:
            point = go.board[i][j]
        else:
            neighbors = go.detect_neighbor(i, j)
            #print((i, j), "neighbors", (i, j) in neighbors) #[go.board[x][y] for (x, y) in neighbors])
            # Neighbors order: top, bottom, left, right
            if not any(go.board[x][y] == 0 for (x, y) in neighbors):
                if all(go.board[x][y] == 1 for (x, y) in neighbors):
                    point = 1
                elif all(go.board[y][y] == 2 for (x, y) in neighbors):
                    point = 2
        return point

    def hasNoNeighbors(self, go, i, j):
        neighbors = go.detect_neighbor(i, j)
        #print('neighbors', [go.board[x][y] for (x,y) in neighbors])
        #print('neighbors', all(go.board[x][y] == 0 for (x, y) in neighbors))
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
    go = GO(N)
    go.set_board(pieceType, previous_board, board)
    player = AlphaBetaPlayer(pieceType, maxDepth=2, maxActions=5)
    # cProfile.run('action = player.get_input(go)')
    action = player.get_input(go)
    print(action)
    writeOutput(action)