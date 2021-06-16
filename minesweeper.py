import numpy as np
import random as rand
import copy

'''
Q Learning 

Make heuristic - be the number of neighbhoring 

Have Gym generate training data - this data influences the best move on the network - probabalisticly. 

NN to determine whether or not this is a good state. 

NN that takes in the 9 neighboring cells of a cell -> Outputs whether the move is good or not. 


Opt of that game - Q-Val on Bombs should be Bad -> Some negative value 
Q-Val -> Number of Neighboring 
R(s) = The current state reward + Best expected utility for a different choice 

R(s) = Value of the square - number of neighborubg bombs 
-> Q-Value = bomb cell or something? 

Could R(S) be the number of cells that were revealed from that move? COuld potentially make it 
more likely to end games faster by vastly reducing things. 


Possible actions - instead of moving l/r/u/d it is moving to different squares - each Q-Val[row][col] represents
the predicted importance of moving to that square. The values of each square are either if it is a bomb 
(Bad value - or the predicted worth of revealing it - the number of squares it can reveal) 


Possible states -> 
    Table of (Known Number of Bombs, Undiscovered cells) -> Q-Val corresponds to this? 
'''


class Minesweeper:
    '''
    Represents a game of minesweeper
    :param user_board - a 2D array of 1's / 0's representing whether the cell has been clicked (1) or not (0)
    :param sol_board - a 2D array where each valid is either the number of neighboring bombs or the bomb itself (-1)
    :param game_over - flag that represents whether the game is currently over - saves computation of calculating
                       whether a bomb is pressed and having to iterate over all cells in grid.
    :param bombs - the number of bombs in the game. Used when determining if game is over
    :param revealed_cells - the number of cells currently revealed - used in determining if game is over
    '''

    def __init__(self, width=3, height=3, bombs=1):
        rand.seed(500)
        '''
        Creates a Minesweeper object to play minesweeper with.
        :param width: The width of the board
        :param height: The height of the board
        :param bombs: Number of bombs you would like on the board
        '''
        self.user_board = np.array([np.zeros(width)] * height)
        self.sol_board = np.array([np.zeros(width)] * height)

        self.game_over = False

        # Uncomment this to have all cells revealed.
        # self.user_board = np.array([np.ones(width)] * height)

        self.bombs = bombs
        self.revealed_cells = 0

        self.generate_bombs()
        self.set_neighboring_bombs()

        self.BOMB_REWARD = -250
        self.EXPLORE_PROB = 0.2
        self.DISCOUNT_FACTOR = 0.8
        self.LEARNING_RATE = 0.01

        self.MAX_MOVES = 1000


    def restart(self):
        return Minesweeper(len(self.sol_board[0]), len(self.sol_board), self.bombs)

    def set_neighboring_bombs(self):
        '''
        Initializes the neighboring bombs for all cells.
        '''
        for row in range(len(self.user_board)):
            for col in range(len(self.user_board[row])):
                out = 0

                if self.sol_board[row][col] != -1:
                    out += self.check_bomb(row - 1, col - 1)
                    out += self.check_bomb(row - 1, col)
                    out += self.check_bomb(row - 1, col + 1)
                    out += self.check_bomb(row, col - 1)
                    out += self.check_bomb(row, col + 1)
                    out += self.check_bomb(row + 1, col - 1)
                    out += self.check_bomb(row + 1, col)
                    out += self.check_bomb(row + 1, col + 1)
                    self.sol_board[row][col] = out

    def check_bomb(self, row, col):
        '''
        Returns 1 if a bomb, zero if not. Used to save time not checking bounds.

        :param row: The row coordinate of the cell
        :param col: The col coordinate of the cell
        :return: 1 if bomb, zero if not.
        '''
        if row < 0 or col < 0 or row > len(self.sol_board) - 1 or col > len(self.sol_board[0]) - 1:
            return 0
        try:
            if self.sol_board[row][col] == -1:
                return 1
            else:
                return 0
        except:
            return 0

    def generate_bombs(self):
        '''
        Generates the bombs for the board
        '''
        bombs_left = self.bombs

        while (bombs_left > 0):
            row = rand.randrange(len(self.user_board))
            col = rand.randrange(len(self.user_board[0]))

            if (self.sol_board[row][col] != -1):
                self.sol_board[row][col] = -1
                bombs_left -= 1

    def reveal(self, row, col):
        '''
        Mutates the player board revealing the cell they pass
        :param row: The row coorindate
        :param col: the col coordinate
        '''
        if row < 0 or col < 0 or row >= len(self.sol_board) or col >= len(self.sol_board[0]):
            raise IndexError("Please enter valid inputs!")

        # if self.user_board[row][col] != 1:
        #     self.revealed_cells += 1
        #
        # self.user_board[row][col] = 1
        self.flood(row, col)
        if self.sol_board[row][col] == -1:
            self.game_over = True

    def flood(self, row, col):
        if self.user_board[row][col] != 1:
            self.revealed_cells += 1
            self.user_board[row][col] = 1

            if self.sol_board[row][col] == 0:
                self.flood_rec(row - 1, col - 1)
                self.flood_rec(row - 1, col)
                self.flood_rec(row - 1, col + 1)

                self.flood_rec(row, col - 1)
                self.flood_rec(row, col + 1)

                self.flood_rec(row + 1, col - 1)
                self.flood_rec(row + 1, col)
                self.flood_rec(row + 1, col + 1)

    def flood_rec(self, row, col):
        if row >= 0 and col >= 0 and row < len(self.sol_board) and col < len(self.sol_board[0]):
            self.flood(row, col)

    def is_game_over(self):
        '''
        Determines whether or not the game is over i.e. whether bomb cell revealed or all non-bomb cells open.
        :return: T/F depending on whether the game is over
        '''
        return self.game_over or self.revealed_cells + self.bombs == len(self.sol_board[0]) * len(self.sol_board)

    def __str__(self):
        out = ""
        for j in range(2 * len(self.user_board[0]) + 1):
            out += "-"
        out += '\n'
        for i in range(len(self.user_board)):

            out += '|'
            for j in range(len(self.user_board[i])):
                if self.user_board[i][j] == 0:
                    out += " |"
                elif self.sol_board[i][j] == -1:
                    out += 'X|'
                else:
                    # out += "{:2f}".format(int(self.sol_board[i][j]))
                    out += str(int(self.sol_board[i][j])).zfill(1) + "|"
            out += '\n'
            for j in range(2 * len(self.user_board[i]) + 1):
                out += "-"
            out += '\n'
        return out

    def possible_states(self):
        '''
        Gets you the possible states - being the ones that are bordering unclicked cells.
        :return: A list of the un-revealed cells that are bordering the revealed cells.
        '''
        out = set()
        for row in range(len(self.user_board)):
            for col in range(len(self.user_board[row])):
                if self.user_board[row][col] == 0:
                    if self.is_revealed(row - 1, col - 1) or \
                            self.is_revealed(row - 1, col) or \
                            self.is_revealed(row - 1, col + 1) or \
                            self.is_revealed(row, col - 1) or \
                            self.is_revealed(row, col + 1) or \
                            self.is_revealed(row + 1, col - 1) or \
                            self.is_revealed(row + 1, col) or \
                            self.is_revealed(row + 1, col + 1):
                        out.add((row, col))
        return list(out)

    def expected_val_for_cell(self, row, col):
        '''
        Calculates the expected state for the cell - being the total sum of all revealed neighboring cells.
        Returns negative because having a large number would technically be bad.
        :param row: The row of the cell
        :param col:  The col of the cell
        :return: The expected state of the cell
        '''
        out = 0
        vals = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for val in vals:
            r,c = val

            upd_x = row + r
            upd_y = col + c

            if self.is_revealed(upd_x, upd_y):
                out += self.sol_board[upd_x, upd_y]
        return -out



    def is_revealed(self, row, col):
        '''
        Determins if a cell has been revealed or not. False if out of bounds
        :param row: Row of the cell
        :param col: Col of the cell
        :return: T/F if the cell has been revealed or not - default False for out of bounds cell.
        '''
        if row < 0 or col < 0 or row >= len(self.user_board) or col >= len(self.sol_board[0]):
            return False
        else:
            return self.user_board[row][col] == 1

    def start(self):
        '''
        Plays a game until the game ends.
        '''
        print("Welcome to minesweeper!\nRow Indices: 1 to ", len(self.sol_board),
              "\nCol Indices: 1 to ", len(self.sol_board[0]))
        while (not self.is_game_over()):
            print(self, "\nPlease enter input (Row Col): \n")
            cleaned = input().split(" ")
            self.reveal(int(cleaned[0]) - 1, int(cleaned[1]) - 1)

        if self.game_over:
            print("You loose :(")
        else:
            print("You win! :) ")

        for i in range(len(self.sol_board)):
            for j in range(len(self.sol_board[i])):
                self.user_board[i][j] = 1
        print(self)


class Policy:

    def __init(self, problem):
        self.best_actions = copy.deepcopy(problem)


def q_solve(problem, iterations):
    pass
    problem = problem.restart()

    # (Tupules of (Known number of neighboring bombs, number of not revealed cells)
    q_vals = {}
    for row in range(10):
        for col in range(10):
            q_vals[(row, col)] = 0

    for c_iter in range(iterations):
        problem = problem.restart()
        row = rand.randrange(0, len(problem.user_board))
        col = rand.randrange(0, len(problem.user_board[0]))

        for c_move in range(problem.MAX_MOVES):
            done, upd_move = q_update(row, col, q_vals, problem)
            if done:
                break
            else:
                row, col = upd_move


def q_update(row, col, q_vals, problem, policy = None):
    pass

#
# def q_solve(problem, iterations):
#     problem = problem.restart() # type: Minesweeper
#
#     q_vals = {}
#     for row in range(len(problem.sol_board)):
#         for col in range(len(problem.sol_board[row])):
#             q_vals[(row,col)] = 0
#
#
#     for c_iter in range(iterations):
#         problem = problem.restart()
#         #May cause problems, but like the game has already been generate so probably not?
#         rand.seed()
#         row = rand.randrange(0, len(problem.user_board))
#         col = rand.randrange(0, len(problem.user_board[0]))
#
#         # print("Starting coords: ", row , " , ", col)
#
#         problem.reveal(row, col)
#         if problem.game_over:
#             q_vals[(row, col)] = problem.BOMB_REWARD
#             continue
#
#         for c_move in range(problem.MAX_MOVES):
#             done, upd_row, upd_col = q_update(row, col, q_vals, problem)
#             if done:
#                 break
#             else:
#                 row = upd_row
#                 col = upd_col
#
#
#     for row in range(len(problem.user_board)):
#         out = ""
#         for col in range(len(problem.user_board[0])):
#             out += str(int(q_vals[(row, col)])) + " "
#         print(out)
#
#     # print(problem.__str__())
#
#
#
#
# def q_update(row, col, q_vals, problem, policy = None):
#     chosen_move = None
#
#     poss_moves = problem.possible_states()
#
#     if rand.random() < problem.EXPLORE_PROB:
#         chosen_move = poss_moves[rand.randrange(0, len(poss_moves))]
#     else:
#         chosen_move = poss_moves[0]
#         best_score = q_vals[chosen_move]
#         for key in poss_moves:
#             if best_score < q_vals[key]:
#                 best_score = q_vals[key]
#                 chosen_move = key
#
#     initial_revealed_cells = problem.revealed_cells
#     q_val_predicted = q_vals[chosen_move]
#
#     upd_row, upd_col = chosen_move
#     problem.reveal(upd_row, upd_col)
#
#     '''
#         What should q-val actual be? The value of moving to that state -
#             - Number of cells it revealed?
#             - The value of the bombs neighboring it?
#             - The utility it provides is the number of cells it reveals so probably more like this value
#     '''
#     q_val_actual = problem.revealed_cells - initial_revealed_cells + 1
#
#
#     if problem.game_over:
#         q_vals[chosen_move] = problem.BOMB_REWARD
#         return True, 0, 0
#     elif problem.is_game_over():
#
#         # print("Game has been won!")
#         #What do we do here?
#         pass
#
#     q_vals[(row,col)] += problem.LEARNING_RATE * ((problem.DISCOUNT_FACTOR * q_val_actual) - q_val_predicted)
#     # q_vals[(chosen_move)] += problem.LEARNING_RATE * (q_val_actual - q_val_predicted)
#     q_vals[chosen_move] = q_val_actual
#     return False, upd_row, upd_col
#
#














if __name__ == "__main__":
    # Minesweeper(6, 6, 3).start()
    game = Minesweeper(6,6, 10)
    q_solve(game, 100000)
    for row in range(len(game.user_board)):
        for col in range(len(game.user_board[0])):
            game.user_board[row][col] = 1
    print(game)


