"""
Usage: RoryTheRobot.py <board_file> <number_of_iterations>

Arguments
    board_file			: name of .yml file from which to read in parameters of the board game
    number_of_iterations	: number of iterations of the board game to run the learning algorithm for

Options
    -h / -help	: displays this help file
"""

# Import all libraries needed
from __future__ import division
import docopt
import yaml
from pylab import *
import numpy as np
import random


test_board = {'Board_dimentions': [3, 3], 'Learning_rate': 0.2, 'Starting_Position': [0, 0], 'Number_of_deaths': 1, 'Success_rate': 0.8, 'Goal_Position': [2, 2], 'Costs': {'Death': -50, 'Goal': 50, 'Move': -1}, 'Discount_rate': 0.5, 'Death_Positions': {'Death 1': [1, 1]}, 'Action_selection_parameter': 0.6}
test_arguments = {'<board_file>': 'board2.yml', '<number_of_iterations>': '500'}



class Squares():
	"""
	A class which holds information about each 'normal' square and starting square on the playing board
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
	"""
	def __init__(self, playing_board, learning_rate, discount_rate, action_selection_parameter, success_rate):
		self.playing_board = playing_board
		self.action_selection_parameter = action_selection_parameter
		self.Vs = self.initialiseVs()
		self.Qs = self.initialiseQs()
		self.current_moves = 0
		self.current_iteration = 0
		self.x_pos = self.playing_board.starting_x
		self.y_pos = self.playing_board.starting_y
		self.learning_rate = learning_rate
		self.discount_rate = discount_rate
		self.transitions = self.find_transitions(success_rate)

	def initialiseVs(self):
		"""
		Initialises the robot's knowledge of the V values of every square

			>>> b = Board(test_board, test_arguments)
			>>> r = b.robot
			>>> r.Vs
			{(0, 1): 0, (1, 2): 0, (0, 0): 0, (2, 1): 0, (0, 2): 0, (2, 0): 0, (2, 2): 0, (1, 0): 0, (1, 1): 0}
		"""
		Vs = {tuple(sqr.coords):0 for row in self.playing_board.board_squares for sqr in row}
		return Vs

	def initialiseQs(self):
		"""
		Initialises the robot's knowledge of the Q values for every action for every square

			>>> b = Board(test_board, test_arguments)
			>>> r = b.robot
			>>> r.Qs
			{(0, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}}
		"""
		Qs = {tuple(sqr.coords):{action:0 for action in ['North', 'East', 'South', 'West']} for row in self.playing_board.board_squares for sqr in row}
		return Qs

	def find_transitions(self, success_rate):
		"""
		Creates the cumulative transitions for every action

			>>> b = Board(test_board, test_arguments)
			>>> r = b.robot
			>>> r.find_transitions(0.7)
			{'West': [0.10000000000000002, 0.10000000000000002, 0.10000000000000002, 0.7], 'East': [0.10000000000000002, 0.7, 0.10000000000000002, 0.10000000000000002], 'North': [0.7, 0.10000000000000002, 0.10000000000000002, 0.10000000000000002], 'South': [0.10000000000000002, 0.10000000000000002, 0.7, 0.10000000000000002]}
		"""
		actions = ['North', 'East', 'South', 'West']
		return {action:[success_rate if action==actions[i] else (1-success_rate)/3 for i in range(4)] for action in actions}

	def choose_random_action(self):
		"""
		Chooses an action uniformly from the available actions for that square

			>>> random.seed(4)
			>>> b = Board(test_board, test_arguments)
			>>> b.robot.choose_random_action()
			'North'
			>>> b.robot.choose_random_action()
			'North'
			>>> b.robot.choose_random_action()
			'East'
			>>> b.robot.choose_random_action()
			'North'
		"""
		return random.choice(['North', 'East', 'South', 'West'])

	def choose_optimal_action(self, x_pos, y_pos):
		"""
		Chooses the action with the highest Q values assosiated with it for the current square

			>>> random.seed(7)
			>>> b = Board(test_board, test_arguments)
			>>> b.robot.Qs = {(0, 0):{'North':5, 'East':-1, 'West':2}, (0, 1):{'North':2, 'East':4, 'South':-3, 'West':11}, (1, 1):{'North':9, 'East':-17, 'South':55, 'West':6}}
			>>> b.robot.choose_optimal_action(1, 1)
			'South'
			>>> b.robot.choose_optimal_action(1, 1)
			'South'
			>>> b.robot.choose_optimal_action(0, 1)
			'West'
			>>> b.robot.choose_optimal_action(0, 0)
			'North'
			>>> b.robot.choose_optimal_action(0, 1)
			'West'
		"""
		sqr = tuple([x_pos, y_pos])
		return str(max(self.Qs[sqr], key=lambda x: self.Qs[sqr][x]))

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
			return self.choose_optimal_action(x_pos, y_pos)
		else:
			return self.choose_random_action()

	def next_square(self, x_pos, y_pos, action):
		"""
		Chooses the new x and y positions after taking and action, according to the faultiness

			>>> random.seed(11)
			>>> b = Board(test_board, test_arguments)
			>>> b.robot.next_square(0, 0, 'North')
			[0, 1]
			>>> b.robot.next_square(0, 0, 'North')
			[0, 1]
			>>> b.robot.next_square(0, 0, 'East')
			[0, 0]
			>>> b.robot.next_square(0, 0, 'East')
			[1, 0]
			>>> b.robot.next_square(0, 0, 'South')
			[0, 0]
			>>> b.robot.next_square(0, 0, 'South')
			[0, 0]
			>>> b.robot.next_square(0, 0, 'West')
			[0, 0]
			>>> b.robot.next_square(0, 0, 'West')
			[0, 0]
		"""
		rnd_num = random.random()
		new_x, new_y = x_pos, y_pos
		sum_p = 0
		for p in range(4):
			sum_p += self.transitions[action][p]
			if rnd_num < sum_p:
				direction = ['North', 'East', 'South', 'West'][p]
				break
		if direction == 'North':
			new_x = x_pos
			new_y = min(y_pos+1, self.playing_board.grid_height-1)
		if direction == 'South':
			new_x = x_pos
			new_y = max(y_pos-1, 0)
		if direction == 'East':
			new_x = min(x_pos+1, self.playing_board.grid_width-1)
			new_y = y_pos
		if direction == 'West':
			new_x = max(x_pos-1, 0)
			new_y = y_pos
		return [new_x, new_y]

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

		>>> b = Board(test_board, test_arguments)
		>>> b.grid_width
		3
		>>> b.grid_height
		3
		>>> b.number_of_iterations
		500
	"""

	def __init__(self, board, arguments):
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

			>>> random.seed(66)
			>>> b = Board(test_board, test_arguments)
			>>> b.number_of_iterations = 5
			>>> b.simulate()
			>>> b.robot.Vs
			{(0, 1): 0.0, (1, 2): 0, (0, 0): 0.0, (2, 1): 8.4, (0, 2): 0.0, (2, 0): 0.0, (2, 2): 0, (1, 0): 0.0, (1, 1): 0}
		"""
		while self.robot.current_iteration < self.number_of_iterations:
			action = self.robot.select_action(self.robot.x_pos, self.robot.y_pos)
			[new_x, new_y] = self.robot.next_square(self.robot.x_pos, self.robot.y_pos, action)
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
