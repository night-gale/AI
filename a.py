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

BOUND_ENCOUNTERED = 0
OPPONENT_CHESS_FOUND = 1
EMPTY_GRID_FOUND = -1

# weights = np.array(
#     [
#         [6.21, 1.88, 12.4, 0.37, 0.37, 12.4, 1.88, 6.21],
#         [1.88, -1.0, -5.45, -1.40, -1.40, -5.45, -1.00, 1.88], 
#         [12.4, -5.45, 0.03, 0.07, 0.07, 0.03, -5.45, 12.4 ],
#         [0.37, -1.40, 0.07, -1.32, -1.32, 0.07, -1.40, 0.37],
#         [0.37, -1.40, 0.07, -1.32, -1.32, 0.07, -1.40, 0.37],
#         [12.4, -5.45, 0.03, 0.07, 0.07, 0.03, -5.45, 12.4 ],
#         [1.88, -1.0, -5.45, -1.40, -1.40, -5.45, -1.00, 1.88], 
#         [6.21, 1.88, 12.4, 0.37, 0.37, 12.4, 1.88, 6.21],
#     ]
# )

class AI(object):
    def __init__(self, chessboard_size, color, time_out, mode='MCTS'):
        self.chessboard_size = chessboard_size 
        self.color = color 
        self.time_out = time_out 
        self.candidate_list = []
        self.mode = mode
        if mode == 'MCTS':
            self.algorithm = MCTS(time_out)
        elif mode == 'random':
            self.algorithm = RD()


    # candidate_list example [(3, 3), (4, 4)]
    # the last entry of candidate_list would be chosen as the decision
    # index range [0, 7], in total 64 position
    def go(self, chessboard):
        self.candidate_list.clear()
        g = Game(chessboard, self.color)
        self.candidate_list = g.getActions()
        return self.algorithm(g, output=self.candidate_list)



class MiniMax:
    pass

class Game:
    SIM_SIZE = 100 
    REAL_SIZE = 100000
    sim_buffer = np.zeros((SIM_SIZE, 8, 8))
    sim_ptr = 0
    real_buffer = np.zeros((REAL_SIZE, 8, 8))
    real_ptr = 0
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
        else:
            self.getActions()
            if len(self.options) == 0:
                return len(Game.__get_options(-self.player_color, self.board, )) == 0
            else:
                return False

    def eval(self, base_player_color):
        result = np.sum(self.board==base_player_color)
        if result > 32:
            return 1.
        elif result == 32:
            return 0.
        else:
            return 0.
        

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
    
    def nextTurnSim(self, action, ):
        new_board = Game.sim_buffer[Game.sim_ptr]
        Game.sim_buffer[Game.sim_ptr] = self.board
        Game.sim_ptr = (Game.sim_ptr + 1) % Game.SIM_SIZE
        if action is None:
            return Game(new_board, -self.player_color)

        for dx, dy in self.options[action][1]:
            Game.__reverse(action, color=self.player_color, chessboard=new_board, direct=(dx, dy))
        new_board[action[0], action[1]] = self.player_color
        
        return Game(new_board, -self.player_color)
    
    def nextTurn(self, action):
        new_board = Game.real_buffer[Game.real_ptr]
        Game.real_buffer[Game.real_ptr] = self.board
        Game.real_ptr = (Game.real_ptr + 1) % Game.REAL_SIZE
        if action is None:
            return Game(new_board, -self.player_color)

        for dx, dy in self.options[action][1]:
            Game.__reverse(action, color=self.player_color, chessboard=new_board, direct=(dx, dy))
        new_board[action[0], action[1]] = self.player_color
        
        return Game(new_board, -self.player_color)
        if action is None: return Game(np.copy(self.board), -self.player_color)
        
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
        chessboard_size = chessboard.shape[0]
        while flag:
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
            return (None, (dx, dy))

    @staticmethod
    def __reverse(choice, color, chessboard, direct):
        x, y = choice
        x += direct[0]
        y += direct[1]
        while chessboard[x, y] != color:
            chessboard[x, y] = color 
            x += direct[0]
            y += direct[1]


class State:
    """
    State class models (board, action) pair \n
    where board is the board of the ${parent.game.board}, action is ${choice to state}
    """
    def __init__(self, game, root=None, choice=None, parent=None, pplayer=None) -> None:
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
        self.pplayer = pplayer

    def getActions(self, ):
        self.actions = self.game.getActions()

    def bestChild(self, explore=True):
        best = [-1, float('-inf')]
        for ii, s in enumerate(self.children):
            tmp = s.getUCB(explore)
            if tmp > best[1]:
                best[0] = ii
                best[1] = tmp
        
        return self.children[best[0]]

    def getUCB(self, eps=1e-5, c=1/math.sqrt(3), explore=True):
        if explore == True:
            if self.N == 0:
                return float('inf')
            return self.val / self.N + c * math.sqrt(2*math.log(self.parent.N) / self.N)
        else:
            return self.val / self.N
    
    def isLeaf(self, ):
        return len(self.children) == 0
    
    def nextState(self, action, base_state):
        new_game = self.game.nextTurn(action)
        new_state = State(new_game, root=base_state, choice=action, 
                            parent=self, pplayer=self.game.player_color)
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
    
    def __call__(self, game: Game, output=None):
        actions = game.getActions()
        if len(actions) == 0:
            return None
        if output is not None:
            output.append(actions[np.random.randint(len(actions))])
        return actions[np.random.randint(len(actions))]

    
class MCTS:
    def __init__(self, time_out):
        self.base_state = None
        self.time_out = time_out
    
    def __call__(self, game: Game, iters=5000, output=None):
        self.base_state = State(game, )
        start_time = time.time_ns()
        max_t_per_iter = 0.0
        for ii in range(iters):
            t1 = time.time_ns()
            # select the optimal leaf node
            node = self.select()
            # rollout to end of the game
            val = self.bit_simulate(node)
            # 
            self.backprop(node, val)

            action = self.base_state.bestChild(False).choice_to_state
            if output is not None and ii % 6 == 0:
                if action is not None:
                    output.append(action)
            t2 = time.time_ns()
            max_t_per_iter = max(0, t2 - t1)
            if (self.time_out - (time.time_ns() - start_time) / 1e9) < 2 * (max_t_per_iter / 1e9):
                if action is not None:
                    output.append(action)
                return ii
        return ii

    

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
            action = state.getRandChoice()
            state.nextState(action, base_state=self.base_state)
        
    def bit_simulate(self, state):
        val, _, _ = BitBoard.simulate(state.game.board, state.game.player_color)
        if self.base_state.game.player_color == COLOR_BLACK:
            return val, state.game
        else:
            return not val, state.game
        

    def simulate(self, state):
        """
        rollout to the terminal state of the game
        """
        game = state.game
        while game.terminate() is not True:
            actions = game.getActions()
            # no action to take
            if len(actions) == 0:
                game = game.nextTurnSim(None)
                continue

            choice = actions[np.random.randint(len(actions))]
            game = game.nextTurnSim(choice)
        
        # evaluate according to base state player color
        return game.eval(self.base_state.game.player_color), game
        

    def backprop(self, state : State, value):
        """
        update the sequence leads to the simulation result
        """
        # update N of parent nodes
        # update val of parent nodes

        # value is the value for base node
        val, final_game = value
        base_color = self.base_state.game.player_color
        while state is not None:
            # value relative to parent state
            # tells about whether the move to current state is valuable to parent Node or not
            state.val += (int(val) if state.pplayer == base_color else int(not val))
            state.N += 1
            state = state.parent

# idea from 
# https://en.wikipedia.org/wiki/Bitboard
class BitBoard:
    @staticmethod
    def Down(b):
        return b << 8
    
    @staticmethod
    def Up(b):
        return b >> 8

    @staticmethod
    def Right(b):
        # and operation for mask
        return (b & 0xfefefefe_fefefefe) >> 1
        
    @staticmethod
    def Left(b):
        # and operation for mask
        return (b & 0x7f7f7f7f_7f7f7f7f) << 1


    @staticmethod
    def getActions(bs, be, ):
        mv = 0
        for func in {BitBoard.Down, BitBoard.Up, BitBoard.Right, BitBoard.Left}:
            mv |= BitBoard.__getActions(bs, be, func)
        
        for f1 in {BitBoard.Down, BitBoard.Up}:
            for f2 in {BitBoard.Right, BitBoard.Left}:
                mv |= BitBoard.__getActions_(bs, be, f1, f2)
        
        return mv & 0xffff_ffff_ffff_ffff
    
    @staticmethod
    def argwhere(bb):
        result = []
        for ii in range(8):
            for jj in range(8):
                if bb & (1 << (ii * 8 + jj)):
                    result.append((ii, jj))
        return result

    @staticmethod 
    def __getActions(bs, be, func):
        # places with no chess stone
        blank = ~(bs | be)

        cap = func(bs) & be
        # move
        # remove `for` loop for efficiency
        cap |= func(cap) & be
        cap |= func(cap) & be
        cap |= func(cap) & be
        cap |= func(cap) & be
        cap |= func(cap) & be

        return (func(cap) & blank)
    
    @staticmethod
    def __getActions_(bs, be, f1, f2):
        # places with no chess stone
        blank = ~(bs | be)

        cap = f1(f2(bs)) & be
        # move
        # remove `for` loop for efficiency
        cap |= f1(f2(cap)) & be
        cap |= f1(f2(cap)) & be
        cap |= f1(f2(cap)) & be
        cap |= f1(f2(cap)) & be
        cap |= f1(f2(cap)) & be

        return (f1(f2(cap)) & blank)


    @staticmethod
    def bitToBoard(bit):
        return np.array([int((bit & 1 << x) != 0) for x in range(64)]).reshape((8, 8))

    @staticmethod
    def boardToBit(board):
        b = board.reshape(64)
        bbit = 0
        wbit = 0
        for ii in range(64):
            bbit |= int(b[ii]==-1) << ii
            wbit |= int(b[ii]==1) << ii
        return {
            -1: bbit,
             1: wbit
        }
    
    @staticmethod 
    def nextTurn(black, white, player_color, mv):
        """
        returns {black, white, player color}
        """
        my, ene = (black, white) if player_color == COLOR_BLACK else (white, black)
        cap = 0

        my |= mv
        for func in {BitBoard.Down, BitBoard.Up, BitBoard.Right, BitBoard.Left}:
            cap = func(mv) & ene
            cap |= func(cap) & ene 
            cap |= func(cap) & ene 
            cap |= func(cap) & ene 
            cap |= func(cap) & ene 
            cap |= func(cap) & ene 
            if (func(cap) & my) != 0:
                my |= cap
                ene &= ~cap
        for f1 in {BitBoard.Down, BitBoard.Up}:
            for f2 in {BitBoard.Right, BitBoard.Left}:
                cap = f1(f2(mv)) & ene
                cap |= f1(f2(cap)) & ene
                cap |= f1(f2(cap)) & ene
                cap |= f1(f2(cap)) & ene
                cap |= f1(f2(cap)) & ene
                cap |= f1(f2(cap)) & ene
                if (f1(f2(cap)) & my) != 0:
                    my |= cap 
                    ene &= ~cap
        
        if player_color == COLOR_BLACK:
            return my, ene, -player_color
        else:
            return ene, my, -player_color
    
    @staticmethod
    def mvToArray(mv):
        result = []
        for ii in range(64):
            tmp = 1 << ii
            if mv & (tmp):
                result.append(tmp)
        
        return result
    
    @staticmethod 
    def randNextTurn(black, white, player_color, ):
        moves = None
        if player_color == COLOR_BLACK:
            moves = BitBoard.getActions(black, white)
        else:
            moves = BitBoard.getActions(white, black)
        if moves  == 0:
            # no move to take
            return (black, white, -player_color), True
        mv = moves
        moves = BitBoard.mvToArray(moves)
        mv = moves[random.randint(0, len(moves)-1)]
        return BitBoard.nextTurn(black, white, player_color, mv), False
    
    @staticmethod
    def terminate(bs, be):
        return ((bs | be) == 0xffff_ffff_ffff_ffff) \
                or ((bs == 0) or (be == 0))
    
    @staticmethod
    def winner(black):
        bn = bin(black&0xffff_ffff_ffff_ffff).count("1")
        return bn > 32

    @staticmethod
    def simulate(board, player_color):
        bitboard = BitBoard.boardToBit(board)
        black = bitboard[-1]
        white = bitboard[1]
        flag = False
        tpc = player_color
        while not BitBoard.terminate(black, white):
            (black, white, tpc), tmp = BitBoard.randNextTurn(black, white, tpc )
            if tmp == True:
                if flag == True:
                    break
            flag = tmp

        result = BitBoard.winner(black)
        return result, black, white
            
def testBitBoard():
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
    b = BitBoard.boardToBit(__board)
    black = b[-1]
    white = b[1]
    flag = False
    tpc = 1
    b = __board
    while BitBoard.terminate(black, white) is not True:
        plt.subplot(221)
        plot(b)
        (black, white, tpc), tmp = BitBoard.randNextTurn(black, white, tpc )
        if tmp == True:
            if flag == True:
                break
        flag = tmp
        b = BitBoard.bitToBoard(black) * (-1) + BitBoard.bitToBoard(white) * 1
        plt.subplot(222)
        plot(b)
        plt.show()

# if __name__ == "__main__":
    # import matplotlib.pyplot as plt 
    # import matplotlib.patches as patch
    # from matplotlib.collections import PathCollection

    # def plot(board: np.ndarray):
        # h, w = board.shape
        # patches = []
        # colors = []
        # radius = 1. / h / 2
        # for ii in range(board.shape[0]):
            # for jj in range(board.shape[1]):
                # color = (1, 0, 0) if board[ii][jj] == 1 else (0, 0, 1) if board[ii][jj] == -1 else (1, 1, 1)
                # circle = plt.Circle((ii * radius * 2 + radius, jj * radius * 2 + radius), radius, color=color)
                # plt.gca().add_artist(circle)

    # testBitBoard()
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
    ai = AI(__board.shape[0], 1, 5, mode='MCTS')
    bi = AI(__board.shape[0], -1, 5, mode='random')
    game = Game(__board, 1)
    turn_of = 1
    action = None

    plt.ion()
    while game.terminate() is not True:
        if turn_of == 1:
            start = time.time_ns()
            r = ai.go(game.board, )
            end = time.time_ns()
            print("decision took: {} second(s) ".format((end - start) / 1e9))
            print("Searched for {} iterations".format(r))
            turn_of = -turn_of
            action = ai.candidate_list[-1] if len(ai.candidate_list) != 0 else None
        else:
            bi.go(game.board, )
            turn_of = -turn_of
            action = bi.candidate_list[-1] if len(bi.candidate_list) != 0 else None
        start = time.time_ns()
        game.getActions()
        end = time.time_ns()
        print("getActions took: ", (start - end) / 1e9)
        game = game.nextTurn(action)
        # plot(game.board)
        # plt.pause(1)
    # plt.show()

    result = np.sum(game.board==ai.color)
    if result > 32:
        print("AI wins")
    elif result == 32:
        print("Draw")
    else:
        print("AI Loses")