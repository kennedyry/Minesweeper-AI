import numpy as np
import random as rand

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

    def __init__(self, width = 3, height = 3, bombs = 1):
        '''
        Creates a Minesweeper object to play minesweeper with.
        :param width: The width of the board
        :param height: The height of the board
        :param bombs: Number of bombs you would like on the board
        '''
        self.user_board = np.array([np.zeros(width)] * height)
        self.sol_board = np.array([np.zeros(width)] * height)

        self.game_over = False


        self.user_board = np.array([np.ones(width)] * height)


        self.bombs = bombs
        self.revealed_cells = 0

        self.generate_bombs()
        self.set_neighboring_bombs()



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

        while(bombs_left > 0):
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
        if self.user_board[row][col] != 1:
            self.revealed_cells += 1

        self.user_board[row][col] = 1
        if self.sol_board[row][col] == -1:
            self.game_over = True

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

    def start(self):
        '''
        Plays a game until the game ends.
        '''
        print("Welcome to minesweeper!\nRow Indices: 1 to ", len(self.sol_board),
              "\nCol Indices: 1 to ", len(self.sol_board[0]))
        while(not self.is_game_over()):
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

if __name__ == "__main__":
    Minesweeper(5,5,10).start()
