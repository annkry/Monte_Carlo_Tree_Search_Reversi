# algorithm Monte Carlo Tree Search

from cmath import inf
from operator import truediv
import random
import sys
import time
import copy
from collections import defaultdict as dd
import math
# from turtle import *

#####################################################
# turtle graphic
#####################################################
# tracer(0,1)

BOK = 50
SX = -100
SY = 0
M = 8
MAX_DEPTH = 4  # 1
A = 1  # a number of figures
BB = 161  # a number of corners occupied
C = 5  # a number of possible moves
D = 61  # a number of pawns connected with corner pieces
E = 1  # number of glued blank fields
F = 1  # number of pawns glued to empty fields
G = 1
# a=41
# b=180
# d=50


#####################################################

def initial_board():
    B = [[None] * M for i in range(M)]
    B[3][3] = 1
    B[4][4] = 1
    B[3][4] = 0
    B[4][3] = 0
    return B


class Board:
    dirs = [(0, 1), (1, 0), (-1, 0), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1)]

    def __init__(self):
        self.board = initial_board()
        self.fields = set()
        self.move_list = []
        self.history = []
        self.l = [2, 2]
        for i in range(M):
            for j in range(M):
                if self.board[i][j] == None:
                    self.fields.add((j, i))

    def draw(self, file):
        for i in range(M):
            res = []
            for j in range(M):
                b = self.board[i][j]
                if b == None:
                    res.append('.')
                elif b == 1:
                    res.append('#')
                else:
                    res.append('o')
            print(''.join(res), file=file)
        #print()

    def moves(self, player):
        res = []
        for (x, y) in self.fields:
            if any(self.can_beat(x, y, direction, player) for direction in Board.dirs):
                res.append((x, y))
        if not res:
            return [None]
        return res

    def can_beat(self, x, y, d, player):
        dx, dy = d
        x += dx
        y += dy
        cnt = 0
        while self.get(x, y) == 1-player:
            x += dx
            y += dy
            cnt += 1
        return cnt > 0 and self.get(x, y) == player

    def get(self, x, y):
        if 0 <= x < M and 0 <= y < M:
            return self.board[y][x]
        return None

    def do_move(self, move, player):
        self.history.append([x[:] for x in self.board])
        self.move_list.append(move)

        if move == None:
            return
        x, y = move
        x0, y0 = move
        self.board[y][x] = player
        self.l[player] += 1
        self.fields -= set([move])
        for dx, dy in self.dirs:
            x, y = x0, y0
            to_beat = []
            x += dx
            y += dy
            while self.get(x, y) == 1-player:
                to_beat.append((x, y))
                x += dx
                y += dy
            if self.get(x, y) == player:
                for (nx, ny) in to_beat:
                    self.board[ny][nx] = player
                    self.l[player] += 1
                    self.l[1 - player] -= 1

    def result(self):
        res = 0
        for y in range(M):
            for x in range(M):
                b = self.board[y][x]
                if b == 0:
                    res -= 1
                elif b == 1:
                    res += 1
        return res

    def utility(self):
        if self.result() > 0:
            return (inf, 0, 0)
        if self.result() == 0:
            return (0, 0, 0)
        if self.result() < 0:
            return (-inf, 0, 0)

    def terminal(self):
        if not self.fields:
            return True
        if len(self.move_list) < 2:
            return False
        return self.move_list[-1] == self.move_list[-2] == None

    def random_move(self, player):
        ms = self.moves(player)
        if ms:
            return random.choice(ms)
        return [None]

    def cut_off_tests(self, depth):
        if depth > MAX_DEPTH:
            return True
        return False

    def heuristics(self, mov, bit):
        result = self.terminal()
        if result:
            if self.result() > 0:
                return inf
            if self.result() == 0:
                return 0
            if self.result() < 0:
                return -inf
        licz2_1 = 0
        licz2_2 = 0
        rs = A*(self.l[1]-self.l[0])
        corner_1 = 0
        corner_2 = 0
        if self.board[0][0] == 0:
            licz2_1 += 1
            k = 1
            while k <= M - 1 and self.board[0][k] != 1 and self.board[0][k] != None:
                corner_1 += 1
                k += 1
            k = 1
            while k <= M - 1 and self.board[k][0] != 1 and self.board[k][0] != None:
                corner_1 += 1
                k += 1
        if self.board[0][0] == 1:
            licz2_2 += 1
            k = 1
            while k <= M - 1 and self.board[0][k] != 0 and self.board[0][k] != None:
                corner_2 += 1
                k += 1
            k = 1
            while k <= M - 1 and self.board[k][0] != 0 and self.board[k][0] != None:
                corner_2 += 1
                k += 1
        if self.board[0][M - 1] == 0:
            licz2_1 += 1
            k = M - 2
            while k >= 0 and self.board[0][k] != 1 and self.board[0][k] != None:
                corner_1 += 1
                k -= 1
            k = 1
            while k <= M - 1 and self.board[k][M - 1] != 1 and self.board[k][M - 1] != None:
                corner_1 += 1
                k += 1
        if self.board[0][M - 1] == 1:
            licz2_2 += 1
            k = M - 2
            while k >= 0 and self.board[0][k] != 0 and self.board[0][k] != None:
                corner_2 += 1
                k -= 1
            k = 1
            while k <= M - 1 and self.board[k][M - 1] != 0 and self.board[k][M - 1] != None:
                corner_2 += 1
                k += 1
        if self.board[M - 1][0] == 0:
            licz2_1 += 1
            k = M - 2
            while k >= 0 and self.board[k][0] != 1 and self.board[k][0] != None:
                corner_1 += 1
                k -= 1
            k = 1
            while k <= M - 1 and self.board[M - 1][k] != 1 and self.board[M - 1][k] != None:
                corner_1 += 1
                k += 1
        if self.board[M - 1][0] == 1:
            licz2_2 += 1
            k = M - 2
            while k >= 0 and self.board[k][0] != 0 and self.board[k][0] != None:
                corner_2 += 1
                k -= 1
            k = 1
            while k <= M - 1 and self.board[M - 1][k] != 0 and self.board[M - 1][k] != None:
                corner_2 += 1
                k += 1
        if self.board[M - 1][M - 1] == 0:
            licz2_1 += 1
            k = M - 2
            while k >= 0 and self.board[k][M - 1] != 1 and self.board[k][M - 1] != None:
                corner_1 += 1
                k -= 1
            k = M - 2
            while k >= 0 and self.board[M - 1][k] != 1 and self.board[M - 1][k] != None:
                corner_1 += 1
                k -= 1
        if self.board[M - 1][M - 1] == 1:
            licz2_2 += 1
            k = M - 2
            while k >= 0 and self.board[k][M - 1] != 0 and self.board[k][M - 1] != None:
                corner_2 += 1
                k -= 1
            k = M - 2
            while k >= 0 and self.board[M - 1][k] != 0 and self.board[M - 1][k] != None:
                corner_2 += 1
                k -= 1
        dicti1 = {}
        dicti2 = {}
        licz4_1 = 0
        licz4_2 = 0
        licz5_1 = 0
        licz5_2 = 0
        licz6_1 = 0
        licz6_2 = 0
        dire = [[1, 0], [-1, 0], [0, -1], [0, 1],
                [-1, -1], [-1, 1], [1, -1], [1, 1]]
        for i in range(0, M):
            for j in range(0, M):
                if self.board[i][j] == None:
                    stop2 = 1
                    stop1 = 1
                    for d in range(0, 4):
                        xx = i + dire[d][0]
                        yy = j + dire[d][1]
                        if xx >= 0 and xx <= M-1 and yy >= 0 and yy <= M-1:
                            if self.board[xx][yy] == 1:
                                if (xx, yy) not in dicti2:
                                    licz5_2 += 1
                                    dicti2[(xx, yy)] = 1
                                if stop2 == 1:
                                    licz4_2 += 1
                                    stop2 = 0
                            elif self.board[xx][yy] == 0:
                                if (xx, yy) not in dicti1:
                                    licz5_1 += 1
                                    dicti1[(xx, yy)] = 1
                                if stop1 == 1:
                                    licz4_1 += 1
                                    stop1 = 0
        return rs

    def max_alpha_beta(self, alpha, beta, depth, start, war):
        player = comp_num
        mov = self.moves(player)
        if war:
            if mov != [None]:
                (a, b) = mov[random.randrange(0, len(mov))]
                return (-inf, a, b)
            else:
                return (-inf, None, None)

        px = None
        py = None

        if self.terminal():
            return self.utility()
        if self.cut_off_tests(depth):
            return (self.heuristics(mov, player), 0, 0)
        value = -inf
        moves = mov
        if mov != [None]:
            for iter in moves:
                j = iter[-1]
                i = iter[-2]
                x, y = (i, j)
                x0, y0 = (i, j)
                self.board[j][i] = player
                self.l[player] += 1
                self.fields -= set([(i, j)])
                to_beatt = []
                for dx, dy in self.dirs:
                    x, y = x0, y0
                    to_beat = []
                    x += dx
                    y += dy
                    while self.get(x, y) == 1-player:
                        to_beat.append((x, y))
                        x += dx
                        y += dy
                    if self.get(x, y) == player:
                        for (nx, ny) in to_beat:
                            self.board[ny][nx] = player
                            to_beatt.append((nx, ny))
                            self.l[player] += 1
                            self.l[1-player] -= 1
                (m, min_i, in_j) = self.min_alpha_beta(
                    alpha, beta, depth + 1, False)
                if m > value:
                    value = m
                    px = i
                    py = j
                for (nx, ny) in to_beatt:
                    self.board[ny][nx] = 1 - player
                    self.l[1 - player] += 1
                    self.l[player] -= 1
                self.board[j][i] = None
                self.l[player] -= 1
                self.fields.add((i, j))
                if value >= beta:
                    return (value, px, py)

                if value > alpha:
                    alpha = value
        else:
            (m, min_i, in_j) = self.min_alpha_beta(
                alpha, beta, depth + 1, False)
            if m > value:
                value = m
            if value >= beta:
                return (value, px, py)

            if value > alpha:
                alpha = value

        return (value, px, py)

    def min_alpha_beta(self, alpha, beta, depth, start):
        player = play_num
        qx = None
        qy = None
        mov = self.moves(player)
        if self.terminal():
            return self.utility()
        if self.cut_off_tests(depth):
            return (self.heuristics(mov, player), 0, 0)
        minv = inf
        moves = mov
        if mov != [None]:
            for i, j in moves:
                S = self
                x, y = (i, j)
                x0, y0 = (i, j)
                S.board[j][i] = player
                S.l[player] += 1
                S.fields -= set([(i, j)])
                to_beatt = []
                for dx, dy in S.dirs:
                    x, y = x0, y0
                    to_beat = []
                    x += dx
                    y += dy
                    while S.get(x, y) == 1-player:
                        to_beat.append((x, y))
                        x += dx
                        y += dy
                    if S.get(x, y) == player:
                        for (nx, ny) in to_beat:
                            S.board[ny][nx] = player
                            to_beatt.append((nx, ny))
                            S.l[player] += 1
                            S.l[1-player] -= 1
                (m, max_i, max_j) = S.max_alpha_beta(
                    alpha, beta, depth + 1, False, False)
                if m < minv:
                    minv = m
                    qx = i
                    qy = j
                for (nx, ny) in to_beatt:
                    self.board[ny][nx] = 1 - player
                    self.l[1 - player] += 1
                    self.l[player] -= 1
                self.board[j][i] = None
                self.l[player] -= 1
                self.fields.add((i, j))
                if minv <= alpha:
                    return (minv, qx, qy)

                if minv < beta:
                    beta = minv
        else:
            (m, max_i, max_j) = self.max_alpha_beta(
                alpha, beta, depth + 1, False, False)
            if m < minv:
                minv = m
            if minv <= alpha:
                return (minv, qx, qy)

            if minv < beta:
                beta = minv

        return (minv, qx, qy)


BOARD_SIZE = 8
PL_COMP = 1
NUM_COMP = 0
play_num = 0
comp_num = 1
m_time = 0.5
direction = [[0, 1], [1, 1], [1, 0], [1, -1],
             [0, -1], [-1, -1], [-1, 0], [-1, 1]]


def mcts_move(board):
    # node - information included [(x,y), number of games, number of wins, number of visits, children]
    def ucb(node_t, t, const_val):
        name, number_of_plays, win_rew, ile_razy_odw, childrens = node_t

        if number_of_plays == 0:
            number_of_plays = 0.00000000001

        if t == 0:
            t = 1
        return (win_rew / number_of_plays) + const_val * math.sqrt(2 * math.log(t) / number_of_plays)

    def game_play(board_temp, player, depth=0):
        def rozw_plansza(board_temp):
            if board_temp.result() < 0:
                return True
            return False
        if depth > 32:
            return rozw_plansza(board_temp)
        possible_moves = board_temp.moves(player)
        if len(possible_moves) == 0:
            if player == NUM_COMP:
                neg_turn = PL_COMP
            else:
                neg_turn = NUM_COMP
            neg_possible_moves = board_temp.moves(neg_turn)

            if len(neg_possible_moves) == 0:
                return rozw_plansza(board_temp)
            else:
                player = neg_turn
                possible_moves = neg_possible_moves

        temp = possible_moves[random.randrange(0, len(possible_moves))]
        board_temp.do_move(temp, player)

        if player == NUM_COMP:
            player = PL_COMP
        else:
            player = NUM_COMP

        return game_play(board_temp, player, depth=depth + 1)

    def expand(board_temp, player):
        moves = board_temp.moves(player)
        result = []
        for temp in moves:
            result.append((temp, 0, 0, 0, []))
        return result

    # looking for a path through the peaks with the largest ucb calculated
    def find_path(root, total_playout):
        current_path = []
        child = root
        parent_playout = total_playout
        mcts_turn = True

        while True:
            if len(child) == 0:
                break
            maxidxlist = [0]
            cidx = 0
            if mcts_turn:
                maxval = -1
            else:
                maxval = 2

            for n_tuple in child:
                parent, t_playout, win_rew, ile_odw, t_childrens = n_tuple
                if mcts_turn:
                    const_val = ucb(n_tuple, parent_playout, 0.1)
                else:
                    const_val = ucb(n_tuple, parent_playout, -0.1)
                if const_val >= maxval:
                    if const_val == maxval:
                        maxidxlist.append(cidx)
                    else:
                        maxidxlist = [cidx]
                        maxval = const_val
                cidx += 1

            maxidx = maxidxlist[random.randrange(0, len(maxidxlist))]
            parent, t_playout, win_rew, ile_odw, t_childrens = child[maxidx]
            current_path.append(parent)
            parent_playout = t_playout
            child = t_childrens
            mcts_turn = not (mcts_turn)

        return current_path

    root = expand(board, NUM_COMP)
    curr_b = Board()

    for loop in range(0, 5000):
        if (time.time() - start) >= m_time:
            break
        curr_b = copy.deepcopy(board)

        # searches for the current path
        current_path = find_path(root, loop)
        tile = NUM_COMP
        for temp in current_path:
            curr_b.do_move(temp, tile)
            if tile == NUM_COMP:
                tile = PL_COMP
            else:
                tile = NUM_COMP

        # we develop this game and check if we have won
        isWon = game_play(copy.deepcopy(curr_b), tile)
        child = root

        # update of data propagation
        if current_path != [None]:
            for temp in current_path:
                idx = 0
                if temp != None:
                    for n_tuple in child:
                        parent, t_playout, win_rew, ile_odw, t_childrens = n_tuple
                        if parent == None:
                            print(temp, parent)
                        if temp[0] == parent[0] and temp[1] == parent[1]:
                            break
                        idx += 1

                    if temp[0] == parent[0] and temp[1] == parent[1]:
                        t_playout += 1
                        if isWon:
                            win_rew += 1
                        if t_playout >= 5 and len(t_childrens) == 0:
                            t_childrens = expand(curr_b, tile)

                        child[idx] = (parent, t_playout, win_rew,
                                      ile_odw+1, t_childrens)

                    child = t_childrens
        else:
            return (None, None)
    cidx = 0
    maxidxlist = [0]
    max_avg_win_rew = -1
    for n_tuple in root:  # selects traffic with the highest number of visits
        parent, t_playout, win_rew, ile_odw, t_childrens = n_tuple
        if (ile_odw >= max_avg_win_rew):
            if ile_odw == max_avg_win_rew:
                maxidxlist.append(cidx)
            else:
                maxidxlist = [cidx]
                max_avg_win_rew = ile_odw
        cidx += 1
    maxidx = maxidxlist[random.randrange(0, len(maxidxlist))]
    parent, t_playout, win_rew, ile_odw, t_childrens = root[maxidx]
    return parent


player = 0
win = 0
mcts_win = 0
no_choice_0 = False
how_many_moves_per_game = 0
file = open("res.txt", "w")
start1 = time.time()
for it in range(0, 5):
    B = Board()
    no_choice_0 = False
    player = 0
    how_many_moves_per_game = 0
    while True:
        if B.moves(0) == [None] and B.moves(1) == [None]:
            break
        if (player == 0):
            how_many_moves_per_game += 1
            start = time.time()
            if B.moves(0) == [None]:
                no_choice_0 = True
            else:
                no_choice_0 = False
            (qx, qy) = mcts_move(B)
            end = time.time()
            # print("ruch mcts: ",end-start,file=file)
            if qx != None:
                B.do_move((qx, qy), player)
        else:
            start = time.time()
            how_many_moves_per_game += 1
            if no_choice_0:
                (m, qx, qy) = B.max_alpha_beta(-inf, inf, 0, True, True)
            else:
                (m, qx, qy) = B.max_alpha_beta(-inf, inf, 0, True, False)
            end = time.time()
            # print("ruch max: ",end-start,file=file)
            if qx != None:
                B.do_move((qx, qy), player)

        player = 1-player
        if B.terminal():
            print(how_many_moves_per_game, file=file)
            break
    if B.result() < 0:  # win MCTS
        B.draw(file)
        # print("win MCTS - o")
        mcts_win += 1
    if B.result() > 0:
        B.draw(file)
        # print("win MINIMAXA - #")

# NUM_COMP = 0
# PL_COMP = 1
player = 1
for it in range(0, 5):
    player = 1
    no_choice_0 = False
    B = Board()
    how_many_moves_per_game = 0
    while True:
        if B.moves(0) == [None] and B.moves(1) == [None]:
            break
        if (player == 0):
            how_many_moves_per_game += 1
            start = time.time()
            if B.moves(0) == [None]:
                no_choice_0 = True
            else:
                no_choice_0 = False
            (qx, qy) = mcts_move(B)
            end = time.time()
            # print("ruch mcts: ",end-start,file=file)
            if qx != None:
                B.do_move((qx, qy), player)
        else:
            start = time.time()
            how_many_moves_per_game += 1
            if no_choice_0:
                (m, qx, qy) = B.max_alpha_beta(-inf, inf, 0, True, True)
            else:
                (m, qx, qy) = B.max_alpha_beta(-inf, inf, 0, True, False)
            end = time.time()
            # print("ruch max: ",end-start,file=file)
            if qx != None:
                B.do_move((qx, qy), player)

        player = 1-player
        if B.terminal():
            break
    if B.result() < 0:  # win MCTS
        B.draw(file)
        # print("win MCTS - o")
        mcts_win += 1
    if B.result() > 0:
        B.draw(file)
        # print("win minmax - #")
end1 = time.time()
print("MCTS algorithm won " + str(mcts_win) + " game plays.", file=file)
print("time: " + str(end1-start1), file=file)
sys.exit(0)
