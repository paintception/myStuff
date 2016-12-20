import numpy as np
import seaborn
import random

from itertools import izip
from matplotlib import pyplot as plt
from collections import Counter

L = 1001
n_particles = 10000

class RandomWalker():

	def create_walker(): #Creation of 10000 particles in the middle of the list

		walker = [0 for x in range(L)]
		walker[(len(walker)+1)/2] = 1

		return walker
	
	def walking(walker): #A random path for 10 timestamps is created for every particle

		direction_list = []	
		time_steps = list(range(0,11))
		Position_Tracker = []

		for timestep in xrange(0,11):
			direction = random.randint(2,3)
			direction_list.append(direction)

		for t_s,direc in zip(time_steps,direction_list):
			
			if direc == 2: #Go Left
				index_to_move = walker.index(1)
				walker[index_to_move] = 0
				walker[index_to_move-1] = 1
				Position_Tracker.append(walker.index(1))
			
			elif direc == 3: #Go Right
				index_to_move = walker.index(1)
				walker[index_to_move] = 0
				walker[index_to_move+1] = 1
				Position_Tracker.append(walker.index(1))
			
			else:
				raise Exception("Unsupported Direction") 

		return Position_Tracker	

	def data_processer(time_list): #Sorts the different timestamps
		
		l = []
		time_list = np.array(time_list).tolist()
		counter_list = Counter(time_list)

		for key in sorted(counter_list.iterkeys()):
			l.append([key, counter_list[key]])

		return l

	def condition(l,p,t_0):	#Computes the Results for plotting

		positions = [x[0] for x in l]
		amounts = [x[1] for x in l]
		l = zip(positions,amounts)

		num = 0
		den = 0
	
		for i in l:
			num += ((i[0]-t_0)**p)*i[1]
			den += i[1]
			
		r = float(num/den)
		return r		

	if __name__ == '__main__':
		
		Positions_Set = []
		Results = []
		
		for i in xrange(1,n_particles+1):
			original_walker = create_walker()
			P = walking(original_walker)
			Positions_Set.append(P)

		New_Position_Set = []	

		for i in Positions_Set:	#Preparation of the list with positions
								#Every Walker starts in the same position

			if i[0] == 500:
				tmp = [x+1 for x in i]
				New_Position_Set.append(tmp)

			elif i[0] == 502:
				tmp = [x-1 for x in i]
				New_Position_Set.append(tmp)

		New_Position_Set = np.array(New_Position_Set)

		for i in xrange(0,11):	#Computes for every position how many times it has been visited in 1 particular timestamp
			tmp_list = New_Position_Set[:,i]
			sorted_tmp_list = data_processer(tmp_list)
			res_1 = condition(sorted_tmp_list, 2,0)	#Change last argument according i_0, either 0 or 1 
			res_2 = condition(sorted_tmp_list, 1,0)**2 #Change last argument according i_0, either 0 or 1
			final_res = (res_1-res_2)
			Results.append(final_res)
	
	plt.plot(Results,'g')	#Plots
	plt.title('$\langle x^2(t) \\rangle$ - $\langle x(t) \\rangle^2$ as a function of t')
	plt.xlabel('t')
	plt.ylabel('$\langle x^2(t) \\rangle$ - $\langle x(t) \\rangle^2$')
	plt.show()

