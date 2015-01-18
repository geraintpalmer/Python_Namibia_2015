import random

payoff_matrix = {
	'Rock':{'Rock':[0, 0], 'Paper':[-1, 1], 'Scissors':[1, -1]},
	'Paper':{'Rock':[1, -1], 'Paper':[0, 0], 'Scissors':[-1, 1]},
	'Scissors':{'Rock':[-1, 1], 'Paper':[1, -1], 'Scissors':[0, 0]}}
opponents_actions_markov_chain = {
	'Rock':[0.2, 0.3, 0.5],
	'Paper':[0.1, 0.1, 0.8],
	'Scissors':[0.6, 0.4, 0.0]}
number_of_iterations = 100


class Player:
	"""
	A class which holds the player that will learn
	"""
	def __init__(self, action_selection_parameter=0.3, alpha=0.3, beta=0.3):
		"""
		Initialises the player class

			>>> p = Player()
			>>> p.alpha
			0.3
			>>> p.beta
			0.3
			>>> p.action_selection_parameter
			0.3

			>>> q = Player(0.7, 0.2, 0.4)
			>>> q.alpha
			0.2
			>>> q.beta
			0.4
			>>> q.action_selection_parameter
			0.7
		"""
		self.Vs = {'Rock':0, 'Paper':0, 'Scissors':0}
		self.Qs = {'Rock':{'Rock':0, 'Paper':0, 'Scissors':0}, 'Paper':{'Rock':0, 'Paper':0, 'Scissors':0}, 'Scissors':{'Rock':0, 'Paper':0, 'Scissors':0}}
		self.current_state = random.choice(['Rock', 'Paper', 'Scissors'])
		self.action_selection_parameter=action_selection_parameter
		self.alpha = alpha
		self.beta = beta

	def select_action(self):
		"""
		Selects which action to take using the epsilon-soft selection policy

			>>> random.seed(10)
			>>> p = Player(0.6, 0.8, 0.2)
			>>> p.Qs['Rock'] = {'Rock':0.5, 'Paper':2.1, 'Scissors':0.1}
			>>> p.history = 'Rock'
			>>> p.current_state = 'Rock'
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
			return max(self.Qs[self.current_state], key=lambda x: self.Qs[self.current_state][x])
		else:
			return random.choice(['Rock', 'Paper', 'Scissors'])

	def update_Q_and_V(self, reward, action):
		"""
		Updates the Q and V values or a particular state and action pair

			>>> p = Player()
			>>> p.Vs
			{'Scissors': 0, 'Paper': 0, 'Rock': 0}
			>>> p.Qs
			{'Scissors': {'Scissors': 0, 'Paper': 0, 'Rock': 0}, 'Paper': {'Scissors': 0, 'Paper': 0, 'Rock': 0}, 'Rock': {'Scissors': 0, 'Paper': 0, 'Rock': 0}}
			>>> p.current_state = 'Scissors'
			>>> p.update_Q_and_V(1, 'Scissors')
			>>> p.Vs
			{'Scissors': 0.3, 'Paper': 0, 'Rock': 0}
			>>> p.Qs
			{'Scissors': {'Scissors': 0.3, 'Paper': 0, 'Rock': 0}, 'Paper': {'Scissors': 0, 'Paper': 0, 'Rock': 0}, 'Rock': {'Scissors': 0, 'Paper': 0, 'Rock': 0}}
		"""
		self.Qs[self.current_state][action] = (1 - self.alpha)*self.Qs[self.current_state][action] + self.alpha*(reward + self.beta*self.Vs[self.current_state])
		self.Vs[self.current_state] = max(self.Qs[self.current_state].values())

class Opponent():
	"""
	A class holding the opponent
	"""
	def __init__(self, markov_chain):
		"""
		Initialises the oppenent

			>>> o = Opponent(opponents_actions_markov_chain)
			>>> o.markov_chain
			{'Scissors': [0.6, 0.4, 0.0], 'Paper': [0.1, 0.1, 0.8], 'Rock': [0.2, 0.3, 0.5]}
		"""
		self.markov_chain=markov_chain
		self.history = random.choice(['Rock', 'Paper', 'Scissors'])

	def select_action(self):
		"""
		Chooses which action the oppenent will play

			>>> random.seed(55)
			>>> o = Opponent(opponents_actions_markov_chain)
			>>> o.history = 'Rock'
			>>> o.select_action()
			'Scissors'
			>>> o.select_action()
			'Rock'
			>>> o.select_action()
			'Scissors'
			>>> o.select_action()
			'Scissors'
			>>> o.select_action()
			'Scissors'
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
		reward = payoff_matrix[player_action][opponents_action][0]
		player.update_Q_and_V(reward, player_action)
		player.current_state, opponent.history = opponents_action, player_action

	print {action: max(player.Qs[action], key=lambda x: player.Qs[action][x]) for action in ['Rock', 'Paper', 'Scissors']}


