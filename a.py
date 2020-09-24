import numpy as np
import random 
import time
import timeit

from numpy import core 

COLOR_BLACK = -1
COLOR_WHITE = 1 
COLOR_NONE = 0
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

        # TODO
        self.candidate_list = list(self._get_options(chessboard))
        if len(self.candidate_list) == 0:
            return 
    
        #choice = np.random.randint(len(self.candidate_list))
        #self.candidate_list.append(self.candidate_list[choice])


    def min_max(self, depth, chessboard):
        efn = AI._evaluate
        pass
    
    # max search go through all possible choice and return the choices with maximum value
    @staticmethod
    def __max_search(depth, chessboard, color):
        # if max depth reached or reaches end of the game
        if AI.__is_terminate(depth, chessboard):
            return (AI._evaluate(color, chessboard))
        pass
    
    # min search go through all possible choices and returns the choice with minimum value
    @staticmethod
    def __min_search(depth, chessboard, color):
        pass

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
    def __is_terminate(depth, chessboard):
        return depth <= 0 or np.sum(chessboard != COLOR_NONE) == (chessboard.shape[0]) ** 2
        


    @staticmethod
    def __is_stable(cur, chessboard, color):
        """
        cur -- tuple of current index
        color -- the color of the chess to check
        """
        num_stable = 0
        for ii, jj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            pos = AI.__check_direction_state(cur, ( ii,  jj), chessboard, color)
            neg = AI.__check_direction_state(cur, (-ii, -jj), chessboard, color)
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
    def __check_direction_state(cur, direction, chessboard, color):
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


    def _get_options(self, chessboard):
        scores = {}
        result = []
        for ii in range(self.chessboard_size):
            for jj in range(self.chessboard_size):
                if (chessboard[ii][jj] != COLOR_NONE):
                    continue
                for dx in [1, 0, -1]:
                    for dy in [1, 0, -1]:
                        if dy == 0 and dx == 0:
                            continue
                        res = self.__get_next((ii, jj), (dx, dy), chessboard)
                        if res is not False:
                            result.append((ii, jj))
                            # get the number of opponent chesses that are going to be reversed
                            if (ii, jj) in scores.keys():
                                scores[(ii, jj)] += res
                            else:
                                scores[(ii, jj)] = res
        result = zip(scores.keys(), scores.values())
        result = sorted(result, key=lambda x: x[1], reverse=False)

        if len(result) == 0:
            return result
        result, _ = zip(*result)
        return result
                            
    


    def __get_next(self, cur, direction, chessboard): 
        x, y = cur 
        dx, dy = direction
        flag = True
        f1, f2 = None, None
        cnt = 0
        while flag:
            cnt += 1
            x += dx 
            y += dy
            f2 = (x >= 0 and x < self.chessboard_size) \
                 and (y >= 0 and y < self.chessboard_size)
            if not f2:
                break
            f1 = chessboard[x][y] == -self.color
            flag = f1 and f2
        
        # bounds encountered
        if (not f2) or (f2 and chessboard[x][y] == COLOR_NONE):
            return False
        elif (cur[0] == x - dx) and (cur[1] == y - dy):
            # no opponent chess lies in between
            return False
        else:
            return cnt - 1


        
if __name__ == "__main__":
    ai = AI(8, 1, 30)
    board = [
        [ 0,  0,  0,  0,  0,  0,  0,  1, ],
        [ 0,  0,  0,  0,  0,  0,  0,  1, ],
        [ 0,  0,  0,  0,  0,  0,  0,  1, ],
        [ 0,  0,  1,  1,  1,  1,  0,  1, ],
        [ 0,  0,  1,  1,  1,  1,  1,  1, ],
        [ 0,  0,  1,  1,  1,  1,  0,  1, ],
        [ 0,  0,  0,  0,  0,  0,  0,  1, ],
        [ 0,  0,  0,  0,  0,  0,  0,  1, ],
    ]
    board = np.array(board)
    print(ai._evaluate(-1, board))
    print(board)