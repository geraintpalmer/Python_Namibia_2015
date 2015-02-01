import random

payoff_matrix = {
	'Rock':{'Rock':0, 'Paper':-1, 'Scissors':1},
	'Paper':{'Rock':1, 'Paper':0, 'Scissors':-1},
	'Scissors':{'Rock':-1, 'Paper':1, 'Scissors':0}}
opponents_actions_markov_chain = {
	'Rock':[0.2, 0.8, 0.0],
	'Paper':[0.5, 0.0, 0.5],
	'Scissors':[0.1, 0.7, 0.2]}
number_of_iterations = 2000


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
			{'Scissors': 0.1, 'Paper': 0, 'Rock': 0}
			>>> p.Qs
			{'Scissors': {'Scissors': 0.1, 'Paper': 0, 'Rock': 0}, 'Paper': {'Scissors': 0, 'Paper': 0, 'Rock': 0}, 'Rock': {'Scissors': 0, 'Paper': 0, 'Rock': 0}}
		"""
		self.Qs[self.state][action] = (1 - self.learning_rate)*self.Qs[self.state][action] + self.learning_rate*(reward + self.discount_rate*self.Vs[self.state])
		self.Vs[self.state] = max(self.Qs[self.state].values())

class Opponent():
	"""
	A class holding the opponent
	"""
	def __init__(self, markov_chain):
		"""
		Initialises the oppenent

			>>> opponents_actions_markov_chain = {'Rock':[0.2, 0.8, 0.0], 'Paper':[0.5, 0.0, 0.5], 'Scissors':[0.1, 0.7, 0.2]}
			>>> o = Opponent(opponents_actions_markov_chain)
			>>> o.markov_chain
			{'Scissors': [0.1, 0.7, 0.2], 'Paper': [0.5, 0.0, 0.5], 'Rock': [0.2, 0.8, 0.0]}
		"""
		self.markov_chain=markov_chain
		self.history = random.choice(['Rock', 'Paper', 'Scissors'])

	def select_action(self):
		"""
		Chooses which action the oppenent will play

			>>> random.seed(55)
			>>> opponents_actions_markov_chain = {'Rock':[0.2, 0.8, 0.0], 'Paper':[0.5, 0.0, 0.5], 'Scissors':[0.1, 0.7, 0.2]}
			>>> o = Opponent(opponents_actions_markov_chain)
			>>> o.history = 'Rock'
			>>> o.select_action()
			'Paper'
			>>> o.select_action()
			'Rock'
			>>> o.select_action()
			'Paper'
			>>> o.select_action()
			'Paper'
			>>> o.select_action()
			'Paper'
		"""
		rnd_num = random.random()
		sum_p = 0
		for i in range(3):
			sum_p += self.markov_chain[self.history][i]
			if rnd_num <= sum_p:
				return ['Rock', 'Paper', 'Scissors'][i]


# Simulation
if __name__ == '__main__':
	player = Player()
	opponent = Opponent(opponents_actions_markov_chain)
	for iteration in range(number_of_iterations):
		player_action = player.select_action()
		opponents_action = opponent.select_action()
		reward = payoff_matrix[player_action][opponents_action]
		player.update_Q_and_V(reward, player_action)
		player.state, opponent.history = opponents_action, opponents_action

	print {action: max(player.Qs[action], key=lambda x: player.Qs[action][x]) for action in ['Rock', 'Paper', 'Scissors']}

