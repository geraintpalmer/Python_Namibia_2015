import random
import matplotlib.pyplot as plt

payoff_matrix = {
	'Rock':{'Rock':[0, 0], 'Paper':[-1, 1], 'Scissors':[1, -1]},
	'Paper':{'Rock':[1, -1], 'Paper':[0, 0], 'Scissors':[-1, 1]},
	'Scissors':{'Rock':[-1, 1], 'Paper':[1, -1], 'Scissors':[0, 0]}}
number_of_episodes = 2000


class Player:
	"""
	A class which holds the player that will learn
	"""
	def __init__(self, action_selection_parameter=0.6, learning_rate=0.1, discount_rate=0.6):
		"""
		Initialises the player class

			>>> p = Player()
			>>> p.learning_rate
			0.1
			>>> p.discount_rate
			0.6
			>>> p.action_selection_parameter
			0.6

			>>> q = Player(0.7, 0.2, 0.4)
			>>> q.learning_rate
			0.2
			>>> q.discount_rate
			0.4
			>>> q.action_selection_parameter
			0.7
		"""
		self.Vs = {'Rock':0, 'Paper':0, 'Scissors':0}
		self.Qs = {'Rock':{'Rock':0, 'Paper':0, 'Scissors':0}, 'Paper':{'Rock':0, 'Paper':0, 'Scissors':0}, 'Scissors':{'Rock':0, 'Paper':0, 'Scissors':0}}
		self.state = random.choice(['Rock', 'Paper', 'Scissors'])
		self.action_selection_parameter=action_selection_parameter
		self.learning_rate = learning_rate
		self.discount_rate = discount_rate
		self.Qs_time_series = {'Rock':{'Rock':[], 'Paper':[], 'Scissors':[]}, 'Paper':{'Rock':[], 'Paper':[], 'Scissors':[]}, 'Scissors':{'Rock':[], 'Paper':[], 'Scissors':[]}}

	def select_action(self):
		"""
		Selects which action to take using the epsilon-soft selection policy

			>>> random.seed(10)
			>>> p = Player(0.6, 0.8, 0.2)
			>>> p.Qs['Rock'] = {'Rock':0.5, 'Paper':2.1, 'Scissors':0.1}
			>>> p.history = 'Rock'
			>>> p.state = 'Rock'
			>>> p.select_action()
			'Paper'
			>>> p.select_action()
			'Paper'
			>>> p.select_action()
			'Scissors'
			>>> p.select_action()
			'Rock'
			>>> p.select_action()
			'Rock'
		"""
		rnd_num = random.random()
		if rnd_num < 1 - self.action_selection_parameter:
			return max(self.Qs[self.state], key=lambda x: self.Qs[self.state][x])
		return random.choice(['Rock', 'Paper', 'Scissors'])

	def update_Q_and_V(self, reward, action):
		"""
		Updates the Q and V values or a particular state and action pair

			>>> p = Player()
			>>> p.Vs
			{'Scissors': 0, 'Paper': 0, 'Rock': 0}
			>>> p.Qs
			{'Scissors': {'Scissors': 0, 'Paper': 0, 'Rock': 0}, 'Paper': {'Scissors': 0, 'Paper': 0, 'Rock': 0}, 'Rock': {'Scissors': 0, 'Paper': 0, 'Rock': 0}}
			>>> p.state = 'Scissors'
			>>> p.update_Q_and_V(1, 'Scissors')
			>>> p.Vs
			{'Scissors': 0.3, 'Paper': 0, 'Rock': 0}
			>>> p.Qs
			{'Scissors': {'Scissors': 0.3, 'Paper': 0, 'Rock': 0}, 'Paper': {'Scissors': 0, 'Paper': 0, 'Rock': 0}, 'Rock': {'Scissors': 0, 'Paper': 0, 'Rock': 0}}
		"""
		self.Qs[self.state][action] = (1 - self.learning_rate)*self.Qs[self.state][action] + self.learning_rate*(reward + self.discount_rate*self.Vs[self.state])
		self.Vs[self.state] = max(self.Qs[self.state].values())
		for action in ['Rock', 'Paper', 'Scissors']:
			for state in ['Rock', 'Paper', 'Scissors']:
				self.Qs_time_series[state][action].append(self.Qs[state][action])

	def show_Qs(self, player_num):
		"""
		Produce a graph of the Qs' time series'
		"""
		a = ['Rock', 'Paper', 'Scissors']
		for state in range(3):
			plt.subplot(3, 1, state)
			plt.plot(self.Qs_time_series[a[state]]['Rock'], color='r', label='Rock')
			plt.plot(self.Qs_time_series[a[state]]['Paper'], color = 'g', label='Paper')
			plt.plot(self.Qs_time_series[a[state]]['Scissors'], color = 'b', label='Scissors')
			plt.hlines(y=0, xmin=0, xmax=number_of_episodes)
			plt.xlabel('Episodes')
			plt.ylabel('Q-Values')
			plt.title("Q-Values of Player " + str(player_num) +", state " + a[state])
		plt.show()




# Simulation
if __name__ == '__main__':
	player1 = Player(0.5, 0.01, 0.9)
	player2 = Player(0.5, 0.01, 0.9)
	for episode in range(number_of_episodes):
		player1_action = player1.select_action()
		player2_action = player2.select_action()
		reward_player1 = payoff_matrix[player1_action][player2_action][0]
		reward_player2 = payoff_matrix[player1_action][player2_action][1]
		player1.update_Q_and_V(reward_player1, player1_action)
		player2.update_Q_and_V(reward_player2, player2_action)
		player1.state, player2.state = player2_action, player1_action
	player1.show_Qs(1)
	player2.show_Qs(2)


	print 'Player 1\'s optimal actions are:'
	print {action: max(player1.Qs[action], key=lambda x: player1.Qs[action][x]) for action in ['Rock', 'Paper', 'Scissors']}
	print ' '
	print 'Player 2\'s optimal actions are:'
	print {action: max(player2.Qs[action], key=lambda x: player2.Qs[action][x]) for action in ['Rock', 'Paper', 'Scissors']}

