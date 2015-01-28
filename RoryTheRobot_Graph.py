"""
Usage: RoryTheRobot.py <board_file> <number_of_episodes>

Arguments
    board_file			: name of .yml file from which to read in parameters of the board game
    number_of_episodes	: number of episodes of the board game to run the learning algorithm for

Options
    -h / -help	: displays this help file
"""
from __future__ import division
import docopt
import yaml
from pylab import *
import random

# test board and arguments for doctesting
test_board = {'Board_dimentions': [3, 3], 'Learning_rate': 0.2, 'Starting_Position': [0, 0], 'Number_of_deaths': 1, 'Success_rate': 0.8, 'Goal_Position': [2, 2], 'Costs': {'Death': -50, 'Goal': 50, 'Move': -1}, 'Discount_rate': 0.5, 'Death_Positions': {'Death 1': [1, 1]}, 'Action_selection_parameter': 0.6}
test_arguments = {'<board_file>': 'board2.yml', '<number_of_episodes>': '500'}

class Squares():
	"""
	A class which holds information about each square on the playing board
	"""
	def __init__(self, square_coords, identifier, move_cost, reward):
		"""
		Initialises the square

			>>> s = Squares((3, 4), 'Normal', -1, 0)
			>>> s.coords
			(3, 4)
			>>> s.identifier
			'Normal'
			>>> s.move_cost
			-1
			>>> s.reward
			0
		"""
		self.coords = square_coords
		self.identifier = str(identifier)
		self.reward = reward
		self.move_cost = move_cost


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
			>>> r.moves
			0
			>>> r.current_episode
			0
			>>> r.coords
			(0, 0)
			>>> r.action_selection_parameter
			0.6
			>>> r.Vs
			{(0, 1): 0, (1, 2): 0, (0, 0): 0, (2, 1): 0, (0, 2): 0, (2, 0): 0, (2, 2): 0, (1, 0): 0, (1, 1): 0}
			>>> r.Qs
			{(0, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}}
			>>> r.transitions
			{'West': [0.06666666666666665, 0.06666666666666665, 0.06666666666666665, 0.8], 'East': [0.06666666666666665, 0.8, 0.06666666666666665, 0.06666666666666665], 'North': [0.8, 0.06666666666666665, 0.06666666666666665, 0.06666666666666665], 'South': [0.06666666666666665, 0.06666666666666665, 0.8, 0.06666666666666665]}
			>>> r.movement_dict['North']((1, 2))
			(1, 2)
			>>> r.movement_dict['East']((0, 0))
			(1, 0)
		"""
		self.playing_board = playing_board
		self.actions = ['North', 'East', 'South', 'West']
		self.action_selection_parameter = action_selection_parameter
		self.Vs = {sqr.coords:0 for row in self.playing_board.squares for sqr in row}
		self.Qs = Qs = {sqr.coords:{action:0 for action in self.actions} for row in self.playing_board.squares for sqr in row}
		self.moves = 0
		self.current_episode = 0
		self.coords = tuple(self.playing_board.starting_coords)
		self.learning_rate = learning_rate
		self.discount_rate = discount_rate
		self.transitions = {action:[success_rate if action==self.actions[i] else (1-success_rate)/3 for i in range(4)] for action in self.actions}
		self.movement_dict = {'North':lambda coords: (coords[0], min(coords[1]+1, self.playing_board.grid_height-1)),
								'South':lambda coords: (coords[0], max(coords[1]-1, 0)),
								'East':lambda coords: (min(coords[0]+1, self.playing_board.grid_width-1), coords[1]),
								'West':lambda coords:(max(0, coords[0]-1), coords[1])}

	def select_action(self, sqr):
		"""
		Selects which action to take using the epsilon-soft action selection policy

			>>> random.seed(3)
			>>> b = Board(test_board, test_arguments)
			>>> b.robot.Vs = {(0, 0):7, (0, 1):-1, (1, 1):2}
			>>> b.robot.Qs = {(0, 0):{'North':0, 'South':3, 'West':4}, (0, 1):{'North':6, 'West':-1, 'East':3}, (1, 1):{'South':5, 'West':3, 'East':-2}}
			>>> b.robot.select_action((0, 0))
			'West'
			>>> b.robot.select_action((0, 0))
			'East'
			>>> b.robot.select_action((1, 1))
			'South'
			>>> b.robot.select_action((0, 1))
			'North'
		"""
		rnd_num = random.random()
		if rnd_num < 1 - self.action_selection_parameter:
			return str(max(self.Qs[sqr], key=lambda x: self.Qs[sqr][x]))
		return random.choice(self.actions)

	def find_destination(self, sqr, action):
		"""
		Chooses the new coordinates after taking an action, according to the faultiness

			>>> random.seed(11)
			>>> b = Board(test_board, test_arguments)
			>>> b.robot.find_destination((0, 0), 'North')
			(0, 1)
			>>> b.robot.find_destination((0, 0), 'North')
			(0, 1)
			>>> b.robot.find_destination((0, 0), 'East')
			(0, 0)
			>>> b.robot.find_destination((0, 0), 'East')
			(1, 0)
			>>> b.robot.find_destination((0, 0), 'South')
			(0, 0)
			>>> b.robot.find_destination((0, 0), 'South')
			(0, 0)
			>>> b.robot.find_destination((0, 0), 'West')
			(0, 0)
			>>> b.robot.find_destination((0, 0), 'West')
			(0, 0)
			>>> b.robot.find_destination((0, 2), 'North')
			(0, 2)
		"""
		rnd_num = random.random()
		sum_p, indx = 0, 0
		while rnd_num > sum_p:
			direction = self.actions[indx]
			sum_p += self.transitions[action][indx]
			indx += 1
		return self.movement_dict[direction](sqr)

	def Q_Learning(self, action, reward, sqr, new_sqr):
		"""
		Updates the Q and V values

			>>> b = Board(test_board, test_arguments)
			>>> b.robot.Vs
			{(0, 1): 0, (1, 2): 0, (0, 0): 0, (2, 1): 0, (0, 2): 0, (2, 0): 0, (2, 2): 0, (1, 0): 0, (1, 1): 0}
			>>> b.robot.Qs
			{(0, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}}
			>>> b.robot.Q_Learning('North', 10, (0, 0), (0, 1))
			>>> b.robot.Vs
			{(0, 1): 0, (1, 2): 0, (0, 0): 2.0, (2, 1): 0, (0, 2): 0, (2, 0): 0, (2, 2): 0, (1, 0): 0, (1, 1): 0}
			>>> b.robot.Qs
			{(0, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 0): {'West': 0, 'East': 0, 'North': 2.0, 'South': 0}, (2, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (0, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (2, 2): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 0): {'West': 0, 'East': 0, 'North': 0, 'South': 0}, (1, 1): {'West': 0, 'East': 0, 'North': 0, 'South': 0}}
		"""
		self.Qs[sqr][action] = (
			1-self.learning_rate)*self.Qs[sqr][action] + self.learning_rate*(
			reward + self.discount_rate*self.Vs[new_sqr]
			)
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
			>>> b.number_of_episodes
			500
		"""
		self.grid_width = board['Board_dimentions'][0]
		self.grid_height = board['Board_dimentions'][1]
		[self.squares, self.board_to_show] = self.create_squares(board)
		self.starting_coords = board['Starting_Position']
		self.number_of_episodes = int(arguments['<number_of_episodes>'])
		self.robot = Robot(self, board['Learning_rate'], board['Discount_rate'], board['Action_selection_parameter'], board['Success_rate'])

	def create_squares(self, board):
		"""
		Creates a grid of all board squares

			>>> b = Board(test_board, test_arguments)
			>>> [[sqr.identifier for sqr in row] for row in b.squares]
			[['Normal', 'Normal', 'Normal'], ['Normal', 'Death', 'Normal'], ['Normal', 'Normal', 'Goal']]
			>>> b.board_to_show
			[[' ', ' ', 'G'], [' ', 'D', ' '], ['S', ' ', ' ']]
		"""
		sqrs = []
		sqrs_to_show = []
		for row in range(self.grid_height):
			sqrs.append([])
			sqrs_to_show.append([])
			for col in range(self.grid_width):
				identifier = 'Normal'
				move_cost = 0
				reward = 0
				marker = ' '
				if [col, row] == board['Starting_Position']:
					marker = 'S'
				if [col, row] == board['Goal_Position']:
					identifier = 'Goal'
					move_cost = board['Costs']['Move']
					reward = board['Costs']['Goal']
					marker = 'G'
				if any([[col, row] == coords for coords in board['Death_Positions'].values()]):
					identifier = 'Death'
					move_cost = board['Costs']['Move']
					reward = board['Costs']['Death']
					marker = 'D'
				sqrs[row].append(Squares((col, row), identifier, move_cost, reward))
				sqrs_to_show[row].append(marker)
		sqrs_to_show.reverse()
		return [sqrs, sqrs_to_show]

	def simulate(self):
		"""
		Simulates many episodes of the game while the robots learns the best policies

			>>> random.seed(59)
			>>> b = Board(test_board, test_arguments)
			>>> b.number_of_episodes = 8
			>>> b.simulate()
			Press enter to continue.Simulating .......
			Simulated. Press enter to exit.
			>>> b.robot.Vs
			{(0, 1): 0.0, (1, 2): 0, (0, 0): 0.0, (2, 1): 0, (0, 2): 0.0, (2, 0): 0, (2, 2): 0, (1, 0): -8.32, (1, 1): 0}
		"""

		self.robot.Qs_time_series = {sqr.coords:{action:[] for action in self.robot.actions} for row in self.squares for sqr in row}

		plt.ion()
		self.show_board()
		wait = raw_input('Press enter to continue.')
		print 'Simulating .......'
		while self.robot.current_episode < self.number_of_episodes:
			action = self.robot.select_action(self.robot.coords)
			new_coords = self.robot.find_destination(self.robot.coords, action)
			self.robot.moves += 1

			reward = self.squares[new_coords[1]][new_coords[0]].reward + (
				self.squares[new_coords[1]][new_coords[0]].move_cost * self.robot.moves)
			self.robot.Q_Learning(action, reward, self.robot.coords, new_coords)
			
			self.robot.coords = new_coords

			if (self.squares[new_coords[1]][new_coords[0]].identifier == 'Death' or 
				self.squares[new_coords[1]][new_coords[0]].identifier == 'Goal'):
				self.robot.moves = 0
				self.robot.coords = tuple(self.starting_coords)
				self.robot.current_episode += 1

				for row in range(self.grid_height):
					for col in range(self.grid_width):
						for action in self.robot.actions:
							self.robot.Qs_time_series[(col, row)][action].append(self.robot.Qs[(col, row)][action])

		self.update_results()
		wait = raw_input('Simulated. Press enter to exit.')

	def show_board(self):
		"""
		Creates a visual plot of the board
		"""
		collabs = range(self.grid_width)
		rowlabs = range(self.grid_height)
		rowlabs.reverse()
		the_table = plt.table(cellText=self.board_to_show,
                      colWidths=[0.05] * self.grid_width,
                      rowLabels=rowlabs,
                      colLabels=collabs,
                      loc='center')
		the_table.set_fontsize(32)
		the_table.scale(3, 3)
		plt.draw()

	def update_results(self):
		"""
		Updates the table to show the optimal directions to take for each square
		"""
		collabs = range(self.grid_width)
		rowlabs = range(self.grid_height)
		rowlabs.reverse()
		for row in range(self.grid_height):
			for col in range(self.grid_width):
				if self.squares[self.grid_height-row-1][col].identifier == 'Goal':
					self.board_to_show[row][col] = ('G')
				elif self.squares[self.grid_height-row-1][col].identifier == 'Death':
					self.board_to_show[row][col] = ('D')
				elif max(self.robot.Qs[(col, self.grid_height-row-1)], key=lambda x: self.robot.Qs[(col, self.grid_height-row-1)][x]) == 'North':
					self.board_to_show[row][col] = (u"\u2191")
				elif max(self.robot.Qs[(col, self.grid_height-row-1)], key=lambda x: self.robot.Qs[(col, self.grid_height-row-1)][x]) == 'East':
					self.board_to_show[row][col] = (u"\u2192")
				elif max(self.robot.Qs[(col, self.grid_height-row-1)], key=lambda x: self.robot.Qs[(col, self.grid_height-row-1)][x]) == 'South':
					self.board_to_show[row][col] = (u"\u2193")
				elif max(self.robot.Qs[(col, self.grid_height-row-1)], key=lambda x: self.robot.Qs[(col, self.grid_height-row-1)][x]) == 'West':
					self.board_to_show[row][col] = (u"\u2190")
		the_table = plt.table(cellText=self.board_to_show,
                      colWidths=[0.05] * self.grid_width,
                      rowLabels=rowlabs,
                      colLabels=collabs,
                      loc='center')
		the_table.set_fontsize(32)
		the_table.scale(3, 3)
		plt.draw()

	def show_graphs(self, coords):
		"""
		Plots Q against time for a specific state-action pair
		"""
		fig, ax = plt.subplots()
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		plt.plot(self.robot.Qs_time_series[coords]['North'], color='b', label='North')
		plt.plot(self.robot.Qs_time_series[coords]['East'], color='g', label='East')
		plt.plot(self.robot.Qs_time_series[coords]['South'], color='r', label='South')
		plt.plot(self.robot.Qs_time_series[coords]['West'], color='y', label='West')
		legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.xlabel('Episodes')
		plt.ylabel('Q-Values')
		plt.hlines(y=0, xmin=0, xmax=self.number_of_episodes)
		plt.draw()
		wait = raw_input('Press enter to continue.')




if __name__ == '__main__':
	# Read in board from file
	arguments = docopt.docopt(__doc__)
	board_file = arguments['<board_file>']
	boardfile = open(board_file, 'r')
	board = yaml.load(boardfile)
	boardfile.close()

	BoardGame = Board(board, arguments)
	BoardGame.robot.learning_rate = 0.1
	BoardGame.robot.discount_rate = 0.1
	BoardGame.simulate()
	BoardGame.show_graphs((3, 0))