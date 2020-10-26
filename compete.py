import a as AI
import b as BI
import numpy as np
import time
from a import Game

if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    def plot(board: np.ndarray):
        h, w = board.shape
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
    ai = AI.AI(__board.shape[0], 1, 5, mode='QMCTS')
    bi = BI.AI(__board.shape[0], -1, 5, mode='QMCTS')
    game = Game(__board, 1)
    turn_of = 1
    action = None

    plt.ion()
    while game.terminate() is not True:
        if turn_of == 1:
            start = time.time_ns()
            r = ai.go(game.board, )
            end = time.time_ns()
            print("AI decision took: {} second(s) ".format((end - start) / 1e9))
            print("Searched for {} iterations".format(r))
            turn_of = -turn_of
            action = ai.candidate_list[-1] if len(ai.candidate_list) != 0 else None
        else:
            start = time.time_ns()
            r = bi.go(game.board, )
            end = time.time_ns()
            print("BI decision took: {} second(s) ".format((end - start) / 1e9))
            print("Searched for {} iterations".format(r))
            turn_of = -turn_of
            action = bi.candidate_list[-1] if len(bi.candidate_list) != 0 else None
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
        print("BI wins")


