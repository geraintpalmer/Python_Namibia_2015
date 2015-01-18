"""
Usage: RoryTheRobot.py <board_file> <number_of_iterations>

Arguments
    board_file			: name of .yml file from which to read in parameters of the board game
    number_of_iterations	: number of iterations of the board game to run the learning algorithm for

Options
    -h / -help	: displays this help file
"""
from __future__ import division
import docopt
import yaml
from pylab import *
import numpy as np
import random

# test board and arguments for doctesting
test_board = {'Board_dimentions': [3, 3], 'Learning_rate': 0.2, 'Starting_Position': [0, 0], 'Number_of_deaths': 1, 'Success_rate': 0.8, 'Goal_Position': [2, 2], 'Costs': {'Death': -50, 'Goal': 50, 'Move': -1}, 'Discount_rate': 0.5, 'Death_Positions': {'Death 1': [1, 1]}, 'Action_selection_parameter': 0.6}
test_arguments = {'<board_file>': 'board2.yml', '<number_of_iterations>': '500'}


class Squares():
	"""
	A class which holds information about each square on the playing board
	"""
	def __init__(self, square_coords, identifier, cost_per_move, reward):
		"""
		Initialises the square

			>>> s = Squares([3, 4], 'Normal', -1, 0)
			>>> s.coords
			[3, 4]
			>>> s.identifier
			'Normal'
			>>> s.cost_per_move
			-1
			>>> s.reward
			0
		"""
		self.coords = square_coords
		self.identifier = str(identifier)
		self.reward = reward
		self.cost_per_move = cost_per_move


class Robot():
	"""
	A class to hold the robot
	"""
	def __init__(self, playing_board, learning_rate, discount_rate, action_selection_parameter, success_rate):
		"""
		Initialises the robot

			>>> b = Board(test_board, test_arguments)
			>>> r = b.robot
			>>> r.learning_rate
			0.2
			>>> r.discount_rate
			0.5
			>>> r.current_moves
			0
			>>> r.current_iteration
			0
			>>> r.x_pos
			0
			>>> r.y_pos
			0
			>>> r.action_selection_parameter
			0.6
			>>> r.Vs
			{(0, 1): 0, (1, 2): 0, (0, 0): 0, (2, 1): 0, (0, 2): 0, (2, 0): 0, (2, 2): 0, (1, 0): 0, (1, 1): 0}
			>>> r.Qs
			{(0, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}}
			>>> r.transitions
			{'West': [0.06666666666666665, 0.06666666666666665, 0.06666666666666665, 0.8], 'East': [0.06666666666666665, 0.8, 0.06666666666666665, 0.06666666666666665], 'North': [0.8, 0.06666666666666665, 0.06666666666666665, 0.06666666666666665], 'South': [0.06666666666666665, 0.06666666666666665, 0.8, 0.06666666666666665]}
			>>> r.movement_dict['North'](1, 2)
			(1, 2)
			>>> r.movement_dict['East'](0, 0)
			(1, 0)
		"""
		self.playing_board = playing_board
		self.actions = ['North', 'East', 'South', 'West']
		self.action_selection_parameter = action_selection_parameter
		self.Vs = {tuple(sqr.coords):0 for row in self.playing_board.board_squares for sqr in row}
		self.Qs = Qs = {tuple(sqr.coords):{action:0 for action in self.actions} for row in self.playing_board.board_squares for sqr in row}
		self.current_moves = 0
		self.current_iteration = 0
		self.x_pos = self.playing_board.starting_x
		self.y_pos = self.playing_board.starting_y
		self.learning_rate = learning_rate
		self.discount_rate = discount_rate
		self.transitions = {action:[success_rate if action==self.actions[i] else (1-success_rate)/3 for i in range(4)] for action in self.actions}
		self.movement_dict = {'North':lambda x, y: (x, min(y+1, self.playing_board.grid_height-1)),
								'South':lambda x, y: (x, max(y-1, 0)),
								'East':lambda x, y: (min(x+1, self.playing_board.grid_width-1), y),
								'West':lambda x, y:(max(0, x-1), y)}

	def select_action(self, x_pos, y_pos):
		"""
		Selects which action to take using the epsilon-soft action selection policy

			>>> random.seed(3)
			>>> b = Board(test_board, test_arguments)
			>>> b.robot.Vs = {(0, 0):7, (0, 1):-1, (1, 1):2}
			>>> b.robot.Qs = {(0, 0):{'North':0, 'South':3, 'West':4}, (0, 1):{'North':6, 'West':-1, 'East':3}, (1, 1):{'South':5, 'West':3, 'East':-2}}
			>>> b.robot.select_action(0, 0)
			'West'
			>>> b.robot.select_action(0, 0)
			'East'
			>>> b.robot.select_action(1, 1)
			'South'
			>>> b.robot.select_action(0, 1)
			'North'
		"""
		rnd_num = random.random()
		if rnd_num < 1 - self.action_selection_parameter:
			sqr = tuple([x_pos, y_pos])
			return str(max(self.Qs[sqr], key=lambda x: self.Qs[sqr][x]))
		else:
			return random.choice(['North', 'East', 'South', 'West'])

	def find_destination(self, x_pos, y_pos, action):
		"""
		Chooses the new x and y positions after taking and action, according to the faultiness

			>>> random.seed(11)
			>>> b = Board(test_board, test_arguments)
			>>> b.robot.find_destination(0, 0, 'North')
			(0, 1)
			>>> b.robot.find_destination(0, 0, 'North')
			(0, 1)
			>>> b.robot.find_destination(0, 0, 'East')
			(1, 0)
			>>> b.robot.find_destination(0, 0, 'East')
			(1, 0)
			>>> b.robot.find_destination(0, 0, 'South')
			(0, 0)
			>>> b.robot.find_destination(0, 0, 'South')
			(0, 0)
			>>> b.robot.find_destination(0, 0, 'West')
			(0, 0)
			>>> b.robot.find_destination(0, 0, 'West')
			(0, 0)
		"""
		rnd_num = random.random()
		new_x, new_y = x_pos, y_pos
		sum_p = 0
		for p in range(4):
			sum_p += self.transitions[action][p]
			if rnd_num < sum_p:
				direction = ['North', 'East', 'South', 'West'][p]
				break
		return self.movement_dict[action](x_pos, y_pos)

	def Q_Learning(self, action, reward, x, y, new_x, new_y):
		"""
		Updates the Q and V values

			>>> b = Board(test_board, test_arguments)
			>>> b.robot.Vs
			{(0, 1): 0, (1, 2): 0, (0, 0): 0, (2, 1): 0, (0, 2): 0, (2, 0): 0, (2, 2): 0, (1, 0): 0, (1, 1): 0}
			>>> b.robot.Qs
			{(0, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}}
			>>> b.robot.Q_Learning('North', 10, 0, 0, 0, 1)
			>>> b.robot.Vs
			{(0, 1): 0, (1, 2): 0, (0, 0): 2.0, (2, 1): 0, (0, 2): 0, (2, 0): 0, (2, 2): 0, (1, 0): 0, (1, 1): 0}
			>>> b.robot.Qs
			{(0, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 0): {'West': 0, 'East': 0, 'North': 2.0, 'South': 0}, (2, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}}
		"""
		sqr = tuple([x, y])
		new_sqr = tuple([new_x, new_y])
		self.Qs[sqr][action] = (1-self.learning_rate)*self.Qs[sqr][action] + self.learning_rate*(reward + self.discount_rate*self.Vs[new_sqr])
		self.Vs[sqr] = max(self.Qs[sqr].values())



class Board():
	"""
	A class which holds the playing board
	"""

	def __init__(self, board, arguments):
		"""
		Initialises the playing board

			>>> b = Board(test_board, test_arguments)
			>>> b.grid_width
			3
			>>> b.grid_height
			3
			>>> b.number_of_iterations
			500
		"""
		self.grid_width = board['Board_dimentions'][0]
		self.grid_height = board['Board_dimentions'][1]
		[self.board_squares, self.board_to_show] = self.create_board_squares(board)
		[self.starting_x, self.starting_y] = board['Starting_Position']
		self.number_of_iterations = int(arguments['<number_of_iterations>'])
		self.robot = Robot(self, board['Learning_rate'], board['Discount_rate'], board['Action_selection_parameter'], board['Success_rate'])

	def create_board_squares(self, board):
		"""
		Creates a grid of all board squares

			>>> b = Board(test_board, test_arguments)
			>>> [[sqr.identifier for sqr in row] for row in b.board_squares]
			[['Normal', 'Normal', 'Normal'], ['Normal', 'Death', 'Normal'], ['Normal', 'Normal', 'Goal']]
			>>> b.board_to_show
			array([[ 1.,  0.,  0.],
			       [ 0.,  3.,  0.],
			       [ 0.,  0.,  2.]])
		"""
		sqrs = []
		sqrs_to_show = np.zeros((self.grid_height, self.grid_width))
		for row in range(self.grid_height):
			sqrs.append([])
			for col in range(self.grid_width):
				identifier = 'Normal'
				cost_per_move = 0
				reward = 0
				if [col, row] == board['Starting_Position']:
					sqrs_to_show[row][col] = 1
				if [col, row] == board['Goal_Position']:
					identifier = 'Goal'
					cost_per_move = board['Costs']['Move']
					reward = board['Costs']['Goal']
					sqrs_to_show[row][col] = 2
				if any([[col, row] == coords for coords in board['Death_Positions'].values()]):
					identifier = 'Death'
					cost_per_move = board['Costs']['Move']
					reward = board['Costs']['Death']
					sqrs_to_show[row][col] = 3
				sqrs[row].append(Squares([col, row], identifier, cost_per_move, reward))
		return [sqrs, sqrs_to_show]

	def simulate(self):
		"""
		Simulates many iterations of the game while the robots learns the best policies

			>>> random.seed(67)
			>>> b = Board(test_board, test_arguments)
			>>> b.number_of_iterations = 12
			>>> b.simulate()
			>>> b.robot.Vs
			{(0, 1): 0.0, (1, 2): 0.0, (0, 0): 0.0, (2, 1): 7.4, (0, 2): 0.0, (2, 0): 0.0, (2, 2): 0, (1, 0): 0.0, (1, 1): 0}
		"""
		while self.robot.current_iteration < self.number_of_iterations:
			action = self.robot.select_action(self.robot.x_pos, self.robot.y_pos)
			(new_x, new_y) = self.robot.find_destination(self.robot.x_pos, self.robot.y_pos, action)
			self.robot.current_moves += 1
			reward = self.board_squares[new_y][new_x].reward + (self.board_squares[new_y][new_x].cost_per_move * self.robot.current_moves)
			self.robot.Q_Learning(action, reward, self.robot.x_pos, self.robot.y_pos, new_x, new_y)
			self.robot.x_pos, self.robot.y_pos = new_x, new_y
			if self.board_squares[new_y][new_x].identifier == 'Death' or self.board_squares[new_y][new_x].identifier == 'Goal':
				self.robot.current_moves, self.robot.x_pos, self.robot.y_pos = 0, self.starting_x, self.starting_y
				self.robot.current_iteration += 1

	def show_board(self, squarecol, startcol, goalcol, deathcol):
		"""
		Creates a visual plot of the board
		"""
		cmap = matplotlib.colors.ListedColormap([squarecol, startcol, goalcol, deathcol])
		pcolor(self.board_to_show, edgecolors='k', cmap=cmap)
		show()



if __name__ == '__main__':
	# Read in board from file
	arguments = docopt.docopt(__doc__)
	board_file = arguments['<board_file>']
	boardfile = open(board_file, 'r')
	board = yaml.load(boardfile)
	boardfile.close()

	BoardGame = Board(board, arguments)
	BoardGame.show_board('white', 'green', 'gold', 'red')
	BoardGame.simulate()
	print {coords:max(BoardGame.robot.Qs[coords], key=lambda x: BoardGame.robot.Qs[coords][x]) for coords in BoardGame.robot.Vs}
	print BoardGame.robot.Qs