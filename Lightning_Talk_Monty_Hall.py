# The Monty Hall Problem

import random

num_iterations = 1000

swap_results = []
noswap_results = []


for iteration in range(num_iterations):
	doors = ['car', 'goat', 'goat']
	random.shuffle(doors)
	initial_choice = doors.pop(random.choice([0, 1, 2]))
	doors.pop(doors.index('goat'))
	swap_results.append(doors[0])
	noswap_results.append(initial_choice)

num_cars_swap = sum([x=='car' for x in swap_results])
num_cars_noswap = sum([x=='car' for x in noswap_results])

print 'When swapping we get %s' % num_cars_swap
print 'When not swapping we get %s' % num_cars_noswap