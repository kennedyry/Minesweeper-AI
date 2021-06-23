import numpy as np
import random as rand
import copy

LEARNING_RATE = .6

EXPLORE_PROB = 0.01
DISCOUNT_FACTOR = 0.8
LEARNING_RATE = 0.01
MAX_MOVES = 1000

'''
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
        # rand.seed()
        '''
        Creates a Minesweeper object to play minesweeper with.
        :param width: The width of the board
        :param height: The height of the board
        :param bombs: Number of bombs you would like on the board
        '''
        self.user_board = np.array([np.zeros(width)] * height)
        self.sol_board = np.array([np.zeros(width)] * height)

        self.game_over = False

        self.bombs = bombs
        self.revealed_cells = 0

        self.generate_bombs()
        self.set_neighboring_bombs()


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

    def expected_val_for_cell(self, row, col, total_type):
        '''
        Calculates the expected state for the cell - being the total sum of all revealed neighboring cells.
        Returns negative because having a large number would technically be bad.
        :param row: The row of the cell
        :param col:  The col of the cell
        :param total_type:  Whether we are returning the utility of a cell as the summed
                            amount of total revealed bomb counts of neighboring cells
                            or as a 3x3 grid of values representing each revealed neighboring
                            cells neighboring bomb count or ' ' if not revealed.
        :return: The expected state of the cell
        '''
        sum_revealed_bombs = 0
        non_revealed = 0
        revealed = 0
        vals = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        surrounding_states = [[' ', ' ', ' '],
                              [' ', ' ', ' '],
                              [' ', ' ', ' ']]

        for val in vals:
            r, c = val

            upd_r = row + r
            upd_c = col + c


            if self.is_revealed(upd_r, upd_c):
                sum_revealed_bombs += self.sol_board[upd_r][upd_c]
                surrounding_states[r + 1][c + 1] = self.sol_board[upd_r][upd_c]
                revealed += 1
            else:
                non_revealed += 1


        s_states_out = (tuple(surrounding_states[0]),
                           tuple(surrounding_states[1]),
                           tuple(surrounding_states[2]))

        if total_type:
            return sum_revealed_bombs, non_revealed
        else:
            return s_states_out, non_revealed


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


def q_solve(problem, iterations, total_type):
    '''
    Represents our training program, which given a game of minesweeper and the number of
    iterations to train for, returns the appropriate q-values for that game.
    :param problem: The game of minesweeper to train on - i.e. What configuration of the board
    :param iterations: The number of iterations / games to play for training
    :param total_type:  Whether we are returning the utility of a cell as the summed
                            amount of total revealed bomb counts of neighboring cells
                            or as a 3x3 grid of values representing each revealed neighboring
                            cells neighboring bomb count or ' ' if not revealed.
    :return: Q-Values corresponding to the optimal / bad moves to make given possible states.
    '''
    print("Starting Reinforcement Learning with ", iterations, " Iterations")
    problem = problem.restart()

    q_vals = {}

    mean_length_of_game = 0
    games_won = 0
    games_lost = 0

    for c_iter in range(1, iterations + 1):
        if (c_iter % 1000) == 0:
            print("Current Iteration: ", c_iter, " Mean length of game: ", int(mean_length_of_game / (games_won + games_lost)),
                  " Games Lost: ", games_lost, " Games Won: ", games_won)
            mean_length_of_game = 0
            games_won = 0
            games_lost = 0

        problem = problem.restart()
        rand.seed()

        current_row = rand.randrange(0, len(problem.user_board))
        current_col = rand.randrange(0, len(problem.user_board[0]))

        problem.reveal(current_row, current_col)
        if problem.is_game_over():
            continue
        for c_move in range(MAX_MOVES):
            done, r, c = q_update(current_row, current_col, q_vals, problem, total_type)
            if done:
                mean_length_of_game += c_move + 1
                break
            else:
                current_row, current_col = (r, c)
        if problem.game_over:
            games_lost += 1
        else:
            games_won += 1
    print("Finished Reinforcement Learning")
    return q_vals


def q_update(row, col, q_vals, problem, total_type):
    '''
    Method that selectes each move to make given the current state of the game
    and updates the q-values corresponding to the move made.
    :param row: The row of the move previously made
    :param col: The col of the move previously made
    :param q_vals: The current dictionary of q-values
    :param problem: The state of the game / the current minesweeper game.
    :param total_type:  Whether we are returning the utility of a cell as the summed
                            amount of total revealed bomb counts of neighboring cells
                            or as a 3x3 grid of values representing each revealed neighboring
                            cells neighboring bomb count or ' ' if not revealed.
    :return: Whether the game is over (T/F) and the row and col of the move currently made.
    '''
    chosen_move = None
    util_of_move = None


    poss_moves = problem.possible_states()

    bomb_hidden_pairs = {}

    for move in poss_moves:
        r, c = move
        neighboring_bomb_val, hidden_neighbors = problem.expected_val_for_cell(r, c, total_type)

        if not (neighboring_bomb_val, hidden_neighbors) in bomb_hidden_pairs:
            bomb_hidden_pairs[(neighboring_bomb_val, hidden_neighbors)] = move

    if rand.random() < EXPLORE_PROB:
        chosen_move = poss_moves[rand.randrange(0, len(poss_moves))]
        util_of_move = problem.expected_val_for_cell(chosen_move[0], chosen_move[1], total_type)
    else:
        chosen_move = None
        best_score = -float("inf")

        for key in bomb_hidden_pairs:
            # Each key is (#Known surrounding bombs, hidden_neighbors)
            if key not in q_vals:
                q_vals[key] = 0
            if chosen_move is None or q_vals[key] > best_score:
                chosen_move = bomb_hidden_pairs[key]
                best_score = q_vals[key]
                util_of_move = key
    if util_of_move not in q_vals:
        q_vals[util_of_move] = 0
    q_val_predicted = q_vals[util_of_move]

    problem.reveal(chosen_move[0], chosen_move[1])

    q_val_actual = None
    if problem.game_over:
        q_val_actual = -5
    elif problem.is_game_over():
        # We won so maybe some extra reward?
        q_val_actual = 10
    else:
        # Probably isnt a good thing here, but how do we reward it not being a bomb
        q_val_actual = 10

    q_vals[util_of_move] += LEARNING_RATE * ((DISCOUNT_FACTOR * q_val_actual) - q_val_predicted)
    if problem.is_game_over():
        return True, 0, 0
    return False, chosen_move[0], chosen_move[1]


def test_set_accuracy(q_vals, game, iterations, total_type):
    '''
    Does basically what Q_Update is doing except that it does not actually update Q-Values
    or use explore_probability to potentially make random moves - used purely to
    see how many wins / losses it gets solely making decisions based on the values
    of a corresponding move and its q-values.
    :param q_vals: The Q-Values that have been created from our reinforcement model
    :param game: The configuration of the game in which we are playing.
    :param iterations: The number of games to simulate.
    :param total_type:  Whether we are returning the utility of a cell as the summed
                            amount of total revealed bomb counts of neighboring cells
                            or as a 3x3 grid of values representing each revealed neighboring
                            cells neighboring bomb count or ' ' if not revealed.
    '''
    print("Simulating ", iterations,
          " games without explore probability to test win-rate without forced random moves")

    games_won = 0
    games_lost = 0
    for c_idx in range(iterations):
        game = game.restart()
        row = rand.randrange(0, len(game.sol_board))
        col = rand.randrange(0, len(game.sol_board[0]))

        game.reveal(row, col)

        while game.is_game_over():
            game = game.restart()
            rand.seed()
            row = rand.randrange(0, len(game.sol_board))
            col = rand.randrange(0, len(game.sol_board[0]))

            game.reveal(row, col)

        while (not game.is_game_over()):
            chosen_move = None
            util_of_move = None

            poss_moves = game.possible_states()

            bomb_hidden_pairs = {}

            for move in poss_moves:
                r, c = move
                neighboring_bomb_val, hidden_neighbors = game.expected_val_for_cell(r, c, total_type)

                if not (neighboring_bomb_val, hidden_neighbors) in bomb_hidden_pairs:
                    bomb_hidden_pairs[(neighboring_bomb_val, hidden_neighbors)] = move

            chosen_move = None
            best_score = -float("inf")

            for key in bomb_hidden_pairs:
                if key not in q_vals:
                    q_vals[key] = 0
                if chosen_move is None or q_vals[key] > best_score:
                    chosen_move = bomb_hidden_pairs[key]
                    best_score = q_vals[key]
                    util_of_move = key
            if util_of_move not in q_vals:
                q_vals[util_of_move] = 0
            game.reveal(chosen_move[0], chosen_move[1])
        if game.game_over:
            games_lost += 1
        else:
            games_won += 1
    print("Simulated ", iterations, " Games lost: ", games_lost, " Games Won: ", games_won,
          "Win Rate: ", games_won / (games_lost + games_won))

def simulate_single_game(q_vals, game, total_type):
    '''
    Simulates a single game, where a user can play through a game seeing what move the
    agent is making every time based on the q-values that we have gotten from training
    :param q_vals: The q-values gained from training our agent.
    :param game: The game configuration.
    :param total_type:  Whether we are returning the utility of a cell as the summed
                            amount of total revealed bomb counts of neighboring cells
                            or as a 3x3 grid of values representing each revealed neighboring
                            cells neighboring bomb count or ' ' if not revealed.
    '''
    game = game.restart()
    row = rand.randrange(0, len(game.sol_board))
    col = rand.randrange(0, len(game.sol_board[0]))

    game.reveal(row, col)

    while game.is_game_over():
        game = game.restart()
        rand.seed()
        row = rand.randrange(0, len(game.sol_board))
        col = rand.randrange(0, len(game.sol_board[0]))

        game.reveal(row, col)

    while (not game.is_game_over()):
        input("Continue?")
        print(game.__str__())

        chosen_move = None
        util_of_move = None

        poss_moves = game.possible_states()
        bomb_hidden_pairs = {}

        for move in poss_moves:
            r, c = move
            neighboring_bomb_val, hidden_neighbors = game.expected_val_for_cell(r, c, total_type)

            if not (neighboring_bomb_val, hidden_neighbors) in bomb_hidden_pairs:
                bomb_hidden_pairs[(neighboring_bomb_val, hidden_neighbors)] = move

        if 1 == 1:
            chosen_move = None
            best_score = -float("inf")

            for key in bomb_hidden_pairs:
                if key not in q_vals:
                    q_vals[key] = 0
                if chosen_move is None or q_vals[key] > best_score:
                    chosen_move = bomb_hidden_pairs[key]
                    best_score = q_vals[key]
                    util_of_move = key
        if util_of_move not in q_vals:
            q_vals[util_of_move] = 0

        game.reveal(chosen_move[0], chosen_move[1])
    if game.game_over:
        print("You lost")
    else:
        print("You won!")

if __name__ == "__main__":
    game = Minesweeper(9, 9, 10)
    q_vals = q_solve(game, 5000, False)

    #If you want to step through seeing what decisions the agent makes every time run this
    # simulate_single_game(q_vals, game)

    #If you want to see the accuracy of it playing games without making random moves to train
    #Run this.
    for i in range(10):
        test_set_accuracy(q_vals, game, 2000, False)



