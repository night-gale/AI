import numpy as np
import random 
import time
import math

from numpy import core
from numpy.testing._private.utils import tempdir 

COLOR_BLACK = -1
COLOR_WHITE = 1 
COLOR_NONE = 0

INF = 1e8
random.seed(0)

BOUND_ENCOUNTERED = 0
OPPONENT_CHESS_FOUND = 1
EMPTY_GRID_FOUND = -1

class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size 
        self.color = color 
        self.time_out = time_out 
        self.candidate_list = []


    # candidate_list example [(3, 3), (4, 4)]
    # the last entry of candidate_list would be chosen as the decision
    # index range [0, 7], in total 64 position
    def go(self, chessboard):
        self.candidate_list.clear()

        self.candidate_list = list(AI._get_options(self.color, chessboard))
        if len(self.candidate_list) == 0:
            return 
    
        #choice = np.random.randint(len(self.candidate_list))
        #self.candidate_list.append(self.candidate_list[choice])


    def min_max(self, depth, chessboard):
        return AI.__max_search(depth, chessboard, self.color)
    
    # max search go through all possible choice and return the choices with maximum value
    @staticmethod
    def __max_search(depth, chessboard, color):
        # if max depth reached or reaches end of the game
        if AI._is_terminate(depth, chessboard):
            return (AI._evaluate(color, chessboard), None)
    
        value = -INF
        choice = None

        options = AI._get_options(color, chessboard, mode='keep')
        for opt, direct in options:
            newBoard = np.copy(chessboard)
            newBoard[opt] = color
            AI._reverse(opt, color, newBoard, direct[1])
            tv, _ = AI.__min_search(depth-1, chessboard, -color)
            if tv > value:
                value = tv
                choice = opt

        return (value, choice)

    
    @staticmethod
    def _reverse(choice, color, chessboard, directs):
        for d in directs:
            x, y = choice
            while chessboard[x, y] != color:
                chessboard[x, y] = color 
                x += d[0]
                y += d[1]
            

    # min search go through all possible choices and returns the choice with minimum value
    @staticmethod
    def __min_search(depth, chessboard, color):
        if AI._is_terminate(depth, chessboard):
            return (AI._evaluate(color, chessboard), None)
    
        value = INF
        choice = None

        options = AI._get_options(color, chessboard, mode='keep')
        for opt, direct in options:
            # direct: (s, list of (x, y))
            newBoard = np.copy(chessboard)
            AI._reverse(opt, color, newBoard, direct[1])
            tv, _ = AI.__max_search(depth-1, chessboard, -color)
            if tv < value:
                value = tv
                choice = opt

        return (value, choice)



    # based on calculating stable chess aka those won't be reversed in the future
    # TODO: further evaluation when no stable chess found
    @staticmethod
    def _evaluate(color, chessboard) -> float:
        """
        chessboard -- list
        """
        # number of our stable chess - opponents stable chess
        turns = np.sum(chessboard != 0)
        score = 0
        chessboard_size = chessboard.shape[0]
        
        my_stable = 0
        op_stable = 0
        for ii in range(chessboard_size):
            for jj in range(chessboard_size):
                if chessboard[ii][jj] == COLOR_NONE:
                    continue
                if chessboard[ii][jj] == color:
                    my_stable += AI.__is_stable((ii, jj), chessboard, color)
                else:
                    op_stable += AI.__is_stable((ii, jj), chessboard, -color)
        return my_stable - op_stable
    



    @staticmethod
    def _is_terminate(depth, chessboard):
        return depth <= 0 or np.sum(chessboard != COLOR_NONE) == (chessboard.shape[0]) ** 2
        


    @staticmethod
    def __is_stable(cur, chessboard, color):
        """
        cur -- tuple of current index
        color -- the color of the chess to check
        """
        num_stable = 0
        for ii, jj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            pos = AI._check_direction_state(cur, ( ii,  jj), chessboard, color)
            neg = AI._check_direction_state(cur, (-ii, -jj), chessboard, color)
            if pos in (BOUND_ENCOUNTERED, OPPONENT_CHESS_FOUND) \
                    and neg in (BOUND_ENCOUNTERED, OPPONENT_CHESS_FOUND):
                num_stable += 1
            elif (pos == EMPTY_GRID_FOUND and neg == BOUND_ENCOUNTERED) \
                    or (pos == BOUND_ENCOUNTERED and neg == EMPTY_GRID_FOUND):
                num_stable += 1
        if num_stable == 4:
            # 4 lines all stable
            return 1
        return 0

    

    @staticmethod
    def _check_direction_state(cur, direction, chessboard, color):
        """            
        Check the state of the 
        the color of the chess to check
        """
        dx, dy = direction;
        x, y = cur;
        flag = True
        f1, f2 = None, None
        chessboard_size = chessboard.shape[0]
        while flag:
            x += dx
            y += dy 
            f2 = (x >= 0 and x < chessboard_size) \
                 and (y >= 0 and y < chessboard_size)
            if not f2:
                break;
            f1 = chessboard[x, y] == color
            flag = f1 and f2
        
        if not f2:
            return BOUND_ENCOUNTERED 
        elif chessboard[x][y] == -color:
            return OPPONENT_CHESS_FOUND
        elif chessboard[x][y] == COLOR_NONE:
            return EMPTY_GRID_FOUND


    @staticmethod
    def _get_options(color, chessboard, mode=None):
        scores = {}
        result = []
        chessboard_size = chessboard.shape[0]
        for ii in range(chessboard_size):
            for jj in range(chessboard_size):
                if (chessboard[ii][jj] != COLOR_NONE):
                    continue
                for dx in [1, 0, -1]:
                    for dy in [1, 0, -1]:
                        if dy == 0 and dx == 0:
                            continue
                        res = AI.__get_next(color, (ii, jj), (dx, dy), chessboard)
                        if res is not False:
                            result.append((ii, jj))
                            # get the number of opponent chesses that are going to be reversed
                            if (ii, jj) in scores.keys():
                                scores[(ii, jj)][0] += res[0]
                                scores[(ii, jj)][1].append(res[1])
                            else:
                                scores[(ii, jj)] = [res[0], [res[1]]]
        result = zip(scores.keys(), scores.values())
        result = sorted(result, key=lambda x: x[1][0], reverse=False)

        if len(result) == 0:
            return result
        if mode == 'keep':
            return result
        result, _ = zip(*result)
        return result
                            

    @staticmethod
    def __get_next(color, cur, direction, chessboard): 
        """
        returns False if no available
                (val, (x, y)) if direction available
        """
        x, y = cur 
        dx, dy = direction
        flag = True
        f1, f2 = None, None
        cnt = 0
        chessboard_size = chessboard.shape[0]
        while flag:
            cnt += 1
            x += dx 
            y += dy
            f2 = (x >= 0 and x < chessboard_size) \
                 and (y >= 0 and y < chessboard_size)
            if not f2:
                break
            f1 = chessboard[x][y] == -color
            flag = f1 and f2
        
        # bounds encountered
        if (not f2) or (f2 and chessboard[x][y] == COLOR_NONE):
            return False
        elif (cur[0] == x - dx) and (cur[1] == y - dy):
            # no opponent chess lies in between
            return False
        else:
            return (cnt - 1, (dx, dy))

class MiniMax:
    pass

class Game:
    def __init__(self, board, player_color):
        self.board = board
        self.player_color = player_color

    def getActions(self, ):
        pass
    
    def terminate(self, ):
        pass

    def eval(self, ):
        pass

    def getValue(self, final_game, value):
        """
        return the corresponding value according to the simulated final state
        i.e. check player_color
        """
        pass


class State:
    def __init__(self, game, root=None, choice=None, parent=None) -> None:
        self.choice_to_state = choice
        self.parent = parent
        self.root = root 
        self.game = game
        
        # maintain all the states expanded
        self.children = []

        # backpropagated value
        self.val = 0.0

        # number of times being accessed
        self.N = 0
        self.actions = []
        self.selected_actions = []

    def getActions(self, ):
        self.actions = self.game.getActions()

    def getUCB(self, eps=1e-8):
        return self.val + 2 * math.sqrt(math.log(self.root.N+eps) / (self.N + eps))
    
    def isLeaf(self, ):
        return len(self.children) == 0
    
    def nextState(self, action):
        new_game = self.game.nextTurn(action)
        new_state = State(new_game, root=self.root, choice=action, parent=self)
        self.children.append(new_state)
        self.selected_actions.append(action)
        return new_state

    
    def getRandChoice(self, ):
        a = np.random.choice(self.actions)
        while a in self.selected_actions:
            a = np.random.choice(self.actions)
        return a
    
class MCTS:
    def __init__(self, ):
        self.base_state = None
    
    def __call__(self, state : State, iters=1000):
        self.base_state = state
        for ii in range(iters):
            # select the optimal leaf node
            node = self.select()
            # rollout to end of the game
            val = self.simulate(node)
            # 
            self.backprop(node, val)

    

    def select(self, ):
        """
        select the best child so far 
        """
        cur = self.base_state
        while cur.isLeaf() is not True:
            self.expand(cur)
            best = [-1, float('-inf')]
            for ii, s in enumerate(cur.children):
                tmp = s.getUCB()
                if tmp > best[1]:
                    best[0] = ii
                    best[1] = tmp
            cur = cur.children[best[0]]
        
        return cur


    def expand(self, state, ):
        """
        add a new state to the tree
        """
        # when no child states expanded, randomly select one
        if (len(state.children) == 0) or (len(state.children) != len(state.actions)):
            state.children.append(state.nextState(state.getRandChoice()))
        

    def simulate(self, state):
        """
        rollout to the terminal state of the game
        """
        game = state.game
        while game.terminate() is not True:
            actions = game.getActions()
            game = game.nextTurn(np.random.choice(actions))
        
        return game.eval(state.game.player_color), game
        

    def backprop(self, state, value):
        """
        update the sequence leads to the simulation result
        """
        # update N of parent nodes
        # update val of parent nodes
        val, final_game = value
        while state.parent is not None:
            tmp_val = state.game.getValue(final_game, val)
            if state.val < tmp_val:
                state.val = tmp_val
            state.N += 1
            state = state.parent



        


        
if __name__ == "__main__":
    ai = AI(8, 1, 30)
    board = [
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0, -1,  1,  0,  0,  0, ],
        [ 0,  0,  1,  1, -1,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
    ]
    board = np.array(board)
    ai.go(board)
    print(ai.min_max(5, chessboard=board))
    print(ai.candidate_list)
    print(board)