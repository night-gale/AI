import numpy as np
import random 
import time
import math
import timeit

from numpy import core
from numpy.testing._private.utils import tempdir 

COLOR_BLACK = -1
COLOR_WHITE = 1 
COLOR_NONE = 0

INF = 1e8

BOUND_ENCOUNTERED = 0
OPPONENT_CHESS_FOUND = 1
EMPTY_GRID_FOUND = -1

class AI(object):
    def __init__(self, chessboard_size, color, time_out, mode='MCTS'):
        self.chessboard_size = chessboard_size 
        self.color = color 
        self.time_out = time_out 
        self.candidate_list = []
        self.mode = mode
        if mode == 'MCTS':
            self.algorithm = MCTS()
        elif mode == 'random':
            self.algorithm = RD()


    # candidate_list example [(3, 3), (4, 4)]
    # the last entry of candidate_list would be chosen as the decision
    # index range [0, 7], in total 64 position
    def go(self, chessboard):
        g = Game(chessboard, self.color)
        self.candidate_list = g.getActions()
        self.candidate_list.append(self.algorithm(g, ))



class MiniMax:
    pass

class Game:
    def __init__(self, board: np.ndarray, player_color):
        self.board = board
        self.player_color = player_color
        self.options = None
    

    def getActions(self, ):
        if self.options is None:
            self.options = Game.__get_options(self.player_color, self.board, )
        return list(self.options.keys())

    def terminate(self, ):
        """
        * if no empty places remained \n
        * or current player has no action to take and so does the next player \n
        * the game terminate, return True \n
        * otherwise False
        """
        if np.sum(self.board==0) == 0:
            return True
        elif len(Game.__get_options(self.player_color, self.board, )) == 0:
            return len(Game.__get_options(-self.player_color, self.board, )) == 0

    def eval(self, base_player_color):
        result = np.sum(self.board==base_player_color) - np.sum(self.board==-base_player_color) 
        if result > 0:
            return result / (self.board.shape[0] ** 2)
        elif result == 0:
            return 0.
        else:
            return -1.
        

    def getValue(self, final_game, value):
        """
        return the corresponding value according to the simulated final state \n
        i.e. check player_color
        """
        # naive version
        if final_game.player_color == self.player_color:
            return value
        else:
            return -value
    
    def nextTurn(self, action):
        if action is None:
            return Game(np.copy(self.board), -self.player_color)

        new_board = np.copy(self.board)
        for dx, dy in self.options[action][1]:
            Game.__reverse(action, color=self.player_color, chessboard=new_board, direct=(dx, dy))
        
        return Game(new_board, -self.player_color)

    @staticmethod
    def __get_options(color, chessboard, ) -> dict:
        """
        return {action: numOfReversed, [directions to update]}
        """
        scores = {}
        chessboard_size = chessboard.shape[0]
        for ii in range(chessboard_size):
            for jj in range(chessboard_size):
                if (chessboard[ii][jj] != COLOR_NONE):
                    continue
                for dx in [1, 0, -1]:
                    for dy in [1, 0, -1]:
                        if dy == 0 and dx == 0:
                            continue
                        res = Game.__get_next(color, (ii, jj), (dx, dy), chessboard)
                        if res is not False:
                            # get the number of opponent chesses that are going to be reversed
                            if (ii, jj) in scores.keys():
                                scores[(ii, jj)][0] += res[0]
                                scores[(ii, jj)][1].append(res[1])
                            else:
                                scores[(ii, jj)] = [res[0], [res[1]]]
        
        return scores 
        # result = zip(scores.keys(), scores.values())
        # result = sorted(result, key=lambda x: x[1][0], reverse=False)

        # if len(result) == 0:
        #     return result
        # if mode == 'keep':
        #     return result
        # result, _ = zip(*result)
        # return result
    
    @staticmethod
    def __get_next(color, cur, direction, chessboard): 
        """
        returns False if no available \n
                (val, (x, y)) if direction available \n
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

    @staticmethod
    def __reverse(choice, color, chessboard, direct):
        x, y = choice
        while chessboard[x, y] != color:
            chessboard[x, y] = color 
            x += direct[0]
            y += direct[1]
            if x < 0 or x >= 8 or y < 0 or y >= 8:
                break


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
        self.actions = None
        self.selected_actions = []

    def getActions(self, ):
        self.actions = self.game.getActions()

    def bestChild(self, ):
        best = [-1, float('-inf')]
        for ii, s in enumerate(self.children):
            tmp = s.getUCB()
            if tmp > best[1]:
                best[0] = ii
                best[1] = tmp
        
        return self.children[best[0]]

    def getUCB(self, eps=1e-5, c=1/math.sqrt(2)):
        if self.N == 0:
            return float('inf')
        return self.val / self.N + c * math.sqrt(2*math.log(self.root.N) / self.N)
    
    def isLeaf(self, ):
        return len(self.children) == 0
    
    def nextState(self, action, base_state):
        new_game = self.game.nextTurn(action)
        new_state = State(new_game, root=base_state, choice=action, parent=self)
        self.children.append(new_state)
        self.selected_actions.append(action)
    
    def getRandChoice(self, ):
        if self.actions is None:
            self.getActions()
        if len(self.actions) == 0:
            return None
        a = self.actions[np.random.randint(len(self.actions))]
        while a in self.selected_actions:
            a = self.actions[np.random.randint(len(self.actions))]
        return a

class RD:
    def __init__(self) -> None:
        self.base_state = None
    
    def __call__(self, game: Game, ):
        actions = game.getActions()
        if len(actions) == 0:
            return None
        return actions[np.random.randint(len(actions))]

    
class MCTS:
    def __init__(self, ):
        self.base_state = None
    
    def __call__(self, game: Game, iters=30, output=None):
        self.base_state = State(game, )
        for ii in range(iters):
            # select the optimal leaf node
            node = self.select()
            # rollout to end of the game
            val = self.simulate(node)
            # 
            self.backprop(node, val)

            if output is not None:
                output.append(self.base_state.bestChild().choice_to_state)
        return self.base_state.bestChild().choice_to_state

    

    def select(self, ):
        """
        select the best child so far 
        """
        cur = self.base_state
        while True:
            if cur == self.base_state:
                pass
            elif cur.isLeaf() is True:
                break

            self.expand(cur)
            cur = cur.bestChild()
        
        return cur


    def expand(self, state, ):
        """
        add a new state to the tree
        """
        # when no child states expanded, randomly select one
        if (len(state.children) == 0) or (len(state.children) != len(state.actions)):
            state.nextState(state.getRandChoice(), base_state=self.base_state)
        

    def simulate(self, state):
        """
        rollout to the terminal state of the game
        """
        game = state.game
        while game.terminate() is not True:
            actions = game.getActions()
            # no action to take
            if len(actions) == 0:
                game = game.nextTurn(None)
                continue

            choice = np.random.randint(len(actions))
            game = game.nextTurn(actions[choice])
        
        # evaluate according to base state player color
        return game.eval(self.base_state.game.player_color), game
        

    def backprop(self, state, value):
        """
        update the sequence leads to the simulation result
        """
        # update N of parent nodes
        # update val of parent nodes
        val, final_game = value
        while state is not None:
            state.val += val
            state.N += 1
            state = state.parent



        
if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import matplotlib.patches as patch
    from matplotlib.collections import PathCollection

    def plot(board: np.ndarray):
        h, w = board.shape
        patches = []
        colors = []
        radius = 1. / h / 2
        for ii in range(board.shape[0]):
            for jj in range(board.shape[1]):
                color = (1, 0, 0) if board[ii][jj] == 1 else (0, 0, 1) if board[ii][jj] == -1 else (1, 1, 1)
                circle = plt.Circle((ii * radius * 2 + radius, jj * radius * 2 + radius), radius, color=color)
                plt.gca().add_artist(circle)

            
    __board = [
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0, -1,  1,  0,  0,  0, ],
        [ 0,  0,  0,  1, -1,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
        [ 0,  0,  0,  0,  0,  0,  0,  0, ],
    ]


    __board = np.array(__board)
    ai = AI(__board.shape[0], 1, None, mode='MCTS')
    bi = AI(__board.shape[0], -1, None, mode='random')
    game = Game(__board, 1)
    turn_of = 1
    action = None

    plt.ion()
    while game.terminate() is not True:
        if turn_of == 1:
            start = time.time_ns()
            ai.go(game.board, )
            end = time.time_ns()
            print("decision took: {} second(s) ".format((end - start) / 1e9))
            turn_of = -turn_of
            action = ai.candidate_list[-1]
        else:
            bi.go(game.board, )
            turn_of = -turn_of
            action = bi.candidate_list[-1]
        start = time.time_ns()
        game.getActions()
        end = time.time_ns()
        print("getActions took: ", (start - end) / 1e9)
        game = game.nextTurn(action)
        plot(game.board)
        plt.pause(1)
    plt.show()

    result = np.sum(game.board==ai.color)
    if result > 32:
        print("AI wins")
    elif result == 32:
        print("Draw")
    else:
        print("AI Loses")