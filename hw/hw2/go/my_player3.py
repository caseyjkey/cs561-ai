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
        self.type = 'alphaBeta'
        # Depth before using heuristic function
        self.maxDepth = maxDepth
        # Limit the possible actions for each node
        self.maxActions = maxActions
        self.actions = []
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
        #count = int(open('moves.txt', 'r').read().strip())
        self.turns = 0 #count
        #print('turns', self.turns)
        #with open('moves.txt', 'w') as moves:
        score = self.score(go)
        print(score)
        if (not score[1] and score[2] == go.komi): #or (score[1] == 1 and score[2] == 0):
            #print('new game')
            #moves.write("0")
            return [2,2]
            #else:
                #moves.write(str(count + 1))


        v = self.maxNode(go, [], 0, [-np.inf, None], [np.inf, None])
        a = v[1] if v[1] else None

        result = "PASS"

        if a and go.valid_place_check(a[0][0], a[0][1], self.pieceType):
            result = a[0]
        
        print(result, v)
        
        return result
        
    '''
        Black PASS
        ----------
                
        X X O   
        O O O X   
        O O X   
        O X   X   
        ----------

        Black PASS
        ----------
                
        X X X X   
        X O O O   
            O O X 
            O   O 
        ----------
    '''

    def findActions(self, go, player):        
        # Offense
        singleLiberties = {1: set(), 2: set()}
        enemy = 2 if player == 1 else 1
        # Only good if capturing does not result in one liberty for us (snapbacks)
        for group in self.groups[enemy]:
                liberties = group[1]
                if len(liberties) == 1:
                    singleLiberties[enemy] = singleLiberties[enemy] | liberties
        
        seen = set()

        if singleLiberties[enemy]:
            seen = seen | singleLiberties[enemy]
            return singleLiberties[enemy]
            actions = self.actionsFilter(go, list(singleLiberties[enemy]), player)
            if actions:
                return actions


        for group in self.groups[player]:
            liberties = group[1]
            if len(liberties) == 1:# and liberties not in seen:
                singleLiberties[player] = singleLiberties[player] | liberties


        actions = []
        # Defense
        if singleLiberties[player]:
            seen = seen | singleLiberties[player]
            actions = self.actionsFilter(go, list(singleLiberties[player]), player)
        
        if not actions:
            actions = set()
            for group in self.groups[enemy]:
                actions = actions | group[1]
            #actions -= seen
            seen = seen | actions
            actions = self.actionsFilter(go, list(actions), player)
        
            if not actions:
                actions = [[x, y] for x in range(5) for y in range(5) if '(x, y) not in seen' and go.valid_place_check(x, y, player)]
        return actions

    def actionsFilter(self, go, actions, player):
        #Prevents snapbacks
        for a in list(actions):
            board = deepcopy(go.board)
            goCopy = GO(5)
            goCopy.board = board
            valid = goCopy.place_chess(a[0], a[1], player)
            if not valid:
                actions.remove(a)
                continue
            goCopy.remove_died_pieces(3 - player)
            self.groups = self.findGroups(goCopy)
            #print("a", a)
            if len(self.findLiberties(a[0], a[1], goCopy.board)) <= 1:
                indirectLiberty = False
                neighbors = goCopy.detect_neighbor(a[0], a[1])
                for group in self.groups[player]:
                    liberties = group[1]
                    members = group[2]
                    #print()
                    #print(group)
                    #print(len(liberties), allies)
                    #print("libs", liberties)
                    if len(liberties) > 1 and any(neighbor in members for neighbor in neighbors):
                        #print("yo")
                        indirectLiberty = True
                        break
                if not indirectLiberty:
                    #print("2222222")
                    actions.remove(a)
        '''
        for a in list(actions):
            i, j = a
            # Check if the place already has a piece
            if go.board[i][j] != 0:
                go.visualize_board()
                print(a, 'Invalid placement. There is already a chess in this position.')

                actions.remove(a)
                continue
            
            # Copy the board for testing
            test_go = go.copy_board()
            test_board = test_go.board

            # Check if the place has liberty
            test_board[i][j] = player
            test_go.update_board(test_board)
            if len(self.findLiberties(i, j, test_board)) <= 1:
                actions.remove(a)
                continue

            # If not, remove the died pieces of opponent and check again
            test_go.remove_died_pieces(3 - player)
            if not test_go.find_liberty(i, j):
                print('Invalid placement. No liberty found in this position.')
                actions.remove(a)

            # Check special case: repeat placement causing the repeat board state (KO rule)
            else:
                if go.died_pieces and go.compare_board(self.previous_board, test_go.board):
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                    actions.remove(a)
        return actions #
        '''
        return actions #[a for a in actions if go.valid_place_check(a[0], a[1], player)]

    def findGroups(self, go):
        # {piece: groups [ (numLiberties, set(libertyPoints), set(groupMembers))]}
        groups = {1: [], 2: []}
        board = go.board
        points = [[i // 5, i % 5] for i in range(len(board) * len(board))]
        points = [[i, j] for [i, j] in points if board[i][j]]
        visited = set()
        for (row, col) in points:
            if (row, col) not in visited:
                color = board[row][col]
                libertyPoints = set()
                allies = go.ally_dfs(row, col)
                for ally in allies:
                    visited.add(ally)
                    points = self.findLiberties(ally[0], ally[1], board)
                    libertyPoints = libertyPoints | points
                groups[color].append((len(libertyPoints), libertyPoints, allies))
        return groups

    # Returns score and action tuple
    # Action should be a coordinate tuple
    # alpha and beta are subscriptable (become tuple or score and point)
    def maxNode(self, go, actions, depth, alpha, beta):
        gameOver = self.gameOver(go, self.pieceType)
        if depth >= self.maxDepth or gameOver:
            # score = self.score(go)
            # v = score[self.startType] - score[self.startEnemy]
            # self.turns = self.turns + depth
            v = self.heuristic(go, gameOver, self.pieceType)
            return v, []
        
        v = [-np.inf, []]
        self.groups = self.findGroups(go)
        validActions = self.findActions(go, self.pieceType)
        if len(validActions) > self.maxActions:
            #print(len(validActions))
            validActions = sample(validActions, self.maxActions)
            #print(len(validActions))
        #print("Valids", validActions)
        #if not validActions:
            #print("d", depth, actions)
            #go.visualize_board()
            #print(self.groups)
        for a in validActions: 
            result = self.moveOutcome(go, a, depth, alpha, beta, "min")
            if v[0] < result[0]:
                if result[1] is None:
                    print('bork', v, result)
                    print(result[1])
                v[0] = result[0]
                v[1] = [a] + result[1]
            if beta[0] < v[0]:
                return v
            #print(validActions)
            #print(alphaCopy, v)
            alpha = alpha if alpha[0] > v[0] else v
        if v[0] is -np.inf:
            print('max', validActions, 'v', v)
            go.visualize_board()
        return v
    
    def minNode(self, go, actions, depth, alpha, beta):
        #print(depth)
        gameOver = self.gameOver(go, self.enemy)
        if depth >= self.maxDepth or gameOver:
            # score = self.score(go)
            # v = score[self.startType] - score[self.startEnemy]
            # self.turns = self.turns + depth
            v = self.heuristic(go, gameOver, self.enemy)
            return v, []
        
        v = [np.inf, []]
        self.groups = self.findGroups(go)
        validActions = self.findActions(go, self.enemy)
        if len(validActions) > self.maxActions:
            #print(len(validActions))
            validActions = sample(validActions, self.maxActions)
            #print(len(validActions))
        #if not validActions:
            #print("d", depth, actions)
            #go.visualize_board()
            #print(self.groups)
        for a in validActions:
            result = self.moveOutcome(go, a, depth+1, alpha, beta, "max")
            if result[1] is None:
                print('bork mi', v, result)
                print(result[1])
            if v[0] > result[0]:
                v[0] = result[0]
                v[1] = [a] + result[1]
            if v[0] < alpha[0]:
                return v
            beta = beta if beta[0] < v[0] else v
        if v[0] is np.inf:
            print('min', validActions, 'v', v)
            go.visualize_board()
        return v

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
    def moveOutcome(self, go, action, depth, alpha, beta, level):
        player = self.pieceType if level == "min" else self.enemy
        a = action
        #if self.isValid(go, a, player):
        board = deepcopy(go.board)
        goCopy = GO(5)
        goCopy.board = board
        valid = goCopy.place_chess(a[0], a[1], player)
        goCopy.n_move += 1
        goCopy.remove_died_pieces(3 - player)
        if level == "max":
            return self.maxNode(goCopy, action, depth, alpha, beta)
        else:
            return self.minNode(goCopy, action, depth, alpha, beta)
        #else:
        #    score = -10000 if level == "min" else 10000
         #   return (score, actions)

    def isValid(self, go, action, player):
        (i, j) = action
        valid = True
        enemy = 2 if player == 1 else 1
        if go.board[i][j]:
            valid = False
        elif self.checkPoint(go, i, j) == enemy:
            valid = False
            #print('1')
        elif not go.valid_place_check(i, j, player):
            valid = False
            #print('2')
        #print(valid, action, self.checkPoint(go, i, j), go.valid_place_check(action[0], action[1], player, False, True))
        return valid

    def gameOver(self, go, player):
        if any(go.valid_place_check(a // 5, a % 5, player) for a in range(25)):
            return False
        #for i in range(len(go.board)):
        #    for j in range(len(go.board)):
        #        if go.valid_place_check(i, j, player):
        #            return False
        return True
        
    def score(self, go):
        # :param pieceType: 1('X')-Black or 2('O')-White.
        score = {1: 0, 2: go.komi} #14.75 w/o komi
        for row in range(len(go.board)):
            for col in range(len(go.board)):
                player = go.board[row][col] # #self.checkPoint(go, row, col)
                if player:
                    score[player] += 1
        return score
    
    def checkPoint(self, go, i, j):
        point = 0
        if go.board[i][j]:
            point = go.board[i][j]
        else:
            neighbors = go.detect_neighbor(i, j)
            # Neighbors order: top, bottom, left, right
            for (x, y) in neighbors:
                if point != 0 and point != go.board[x][y]:
                    point = None
                    break
                elif go.board[x][y] and not point:
                    point = go.board[x][y]
                elif point and not go.board[x][y]:
                    point = 0
                    break
        return point

    def hasNoNeighbors(self, go, i, j):
        neighbors = go.detect_neighbor(i, j)
        return all(go.board[x][y] == 0 for (x, y) in neighbors)
                

    def heuristic(self, go, gameOver, player):
        stepCost = 10
        # turns not incrementing correctly
        # TODO: implement using go.board's n_move property
        #print(go.n_move)
        enemy = 2 if player == 1 else 1
        player = self.pieceType
        value = 500 - go.n_move
        if gameOver:
            score = self.score(go)
            v = score[player] - score[enemy]
            return value if v >= 0 else -1 * value 

        groups = self.findGroups(go)
        

        # good chance of winning
        # if there is at least one enemy group with one or less liberties
        if any(group[0] == 1 for group in groups[enemy]):
            # print("good chance winning", action, [group for group in groups[enemy] if group[0] <= 1])
            return value - stepCost
        # good chance of losing
        # if we have more than one group with one or less liberties
        elif sum(1 for group in groups[player] if group[0] <= 1) > 1:
            return -1 * (value - stepCost)

        # this could be returning a positive value when enemy is using heuristic
        # enemy heuristic winning means negative value
        # decent chance of winning
        if self.twoGroupsTwoLibertiesWithOneSharedPoint(enemy, groups):
            return value / 2
        sharedSelfPoints = self.twoGroupsTwoLibertiesWithOneSharedPoint(player, groups)
        # if enemy does not share any of these dangerous points, we decent chance of losing
        if sharedSelfPoints and not any(selfPoint in group[1] for group in groups[enemy] for selfPoint in sharedSelfPoints):
            return -1 * value / 2

        # group score
        enemyGroupsWith2Liberties = sum(group[0] == 2 for group in groups[enemy])
        selfGroupsWith2Liberties = sum(group[0] == 2 for group in groups[player])
        groupScore = enemyGroupsWith2Liberties - selfGroupsWith2Liberties

        # liberty score
        selfLiberties = set()
        sharedSelfLiberties = 0
        for group in groups[player]:
            sharedSelfLiberties += len(selfLiberties & group[1])
            selfLiberties = selfLiberties | group[1]
        enemyLiberties = set()
        sharedEnemyLiberties = 0
        for group in groups[enemy]:
            sharedEnemyLiberties += len(enemyLiberties & group[1])
            enemyLiberties = enemyLiberties | group[1]
        libertyScore = sharedEnemyLiberties - sharedSelfLiberties

        # return a randomly scaled score
        total = groupScore * uniform(0, 1) + libertyScore * uniform(0, 1)
        return total

    def findLiberties(self, i, j, board):
        neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        size = len(board)
        liberties = set()
        for neighbor in neighbors:
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                if 0 == board[neighbor[0]][neighbor[1]]:
                    liberties.add(neighbor)
        
        return liberties
    
    # returns the two groups liberties
    def twoGroupsTwoLibertiesWithOneSharedPoint(self, pieceType, groups):
        libertyCounts = {}
        # check if a piece has two groups with two liberties and one liberty is shared
        for i, group in enumerate(groups[pieceType]):
            if group[0] == 2 and i < len(groups[pieceType]) - 1:
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
    player = AlphaBetaPlayer(pieceType, maxDepth=4, maxActions=3) # TOs for 3,6, NO tos 4,3 (29)
    #cProfile.run('action = player.get_input(go)')
    action = player.get_input(go)
    #print(action)
    writeOutput(action)