import numpy as np
import math
import itertools
import time
import copy
#import seaborn

from itertools import izip
from matplotlib import pyplot as plt

D = 1
L = 1001
d = 0.1
tau = 0.001
m = 1000

class Diffusion():

	def vector_condition_a():
			
		vec_a = [0 for x in range(L)]
		vec_a[0] = 1
			
		return vec_a

	def vector_condition_b():

		vec_b = [0 for x in range(L)]
		vec_b[(len(vec_b)+1)/2] = 1

		return vec_b

	def matrix_condition_a(vector,t):
		
		last_element = vector[-1]

		pair_list = []
		tmp_results = []
		results = []

		Ending_Point = math.e**(-D*t/(2*m*d**2))
		block_a = np.array([[0.5*(1+math.e**(-t*D/(m*d**2))),0.5*(1-math.e**(-t*D/(m*d**2)))], 
							[0.5*(1-math.e**(-t*D/(m*d**2))),0.5*(1+math.e**(-t*D/(m*d**2)))]])

		for i,j in zip(vector[::2],vector[1::2]):
			pair_list.append([i,j])

		pair_list = np.array(pair_list)

		for pair in pair_list:
			tmp_results.append(block_a.dot(pair))
		
		for res in tmp_results:
			results.append(np.array(res).tolist())

		new_vector_a = list(itertools.chain(*results))
		new_vector_a.append(last_element*Ending_Point)

		return new_vector_a

	def matrix_condition_b(vector,t):

		tmp_vector = []
		tmp_results = []
		pair_list = []
		results = []

		Starting_Point = math.e**(-t*D/(m*d**2))
		block_b = np.array([[0.5*(1+math.e**(-2*t*D/(m*d**2))),0.5*(1-math.e**(-2*t*D/(m*d**2)))], 
							[0.5*(1-math.e**(-2*t*D/(m*d**2))),0.5*(1+math.e**(-2*t*D/(m*d**2)))]])

		first_element = vector[0]*Starting_Point

		for i in vector[1:]:
			tmp_vector.append(i)
	
		for i,j in zip(tmp_vector[::2],tmp_vector[1::2]):
			pair_list.append([i,j])

		pair_list = np.array(pair_list)

		for pair in pair_list:
			tmp_results.append(block_b.dot(pair))
		
		for res in tmp_results:
			results.append(np.array(res).tolist())

		new_vector_b = list(itertools.chain(*results))
		new_vector_b = [first_element]+new_vector_b

		return new_vector_b
	
	def process_results(vec_res,p):

		counter_vec = [x for x in range(L+1)]
		results = []

		num = 0
		den = 0
		r = 0

		for i, j in zip(counter_vec,vec_res):
			num += ((i-501)**p)*vec_res[i]
			den += vec_res[i]

		if den != 0:
			r = num/den

		return r

	if __name__ == '__main__':

		vec_a = vector_condition_a()
		vec_b = vector_condition_b()
		final_list = []
		
		n_v_a_2 = vec_b

		res_tracker = []

		for t in xrange(0,11):
			print "Processing Time:", t
			for i in xrange(0,m):
				n_v_a = matrix_condition_a(n_v_a_2,t/D)
				n_v_b = matrix_condition_b(n_v_a,t/D)
				n_v_a_2 = matrix_condition_a(n_v_b,t/D)

			res_tracker.append(n_v_a_2)
		
		for r_t in res_tracker:
			res_1 = (d**2)*(process_results(r_t,2))
			res_2 = (d**1)*(process_results(r_t,1))**2
			_res = (res_1-res_2)
			final_res = (d**-2)*_res
		
			final_list.append(final_res)

		thefile = open('VectorA.txt', 'w')
		for i in final_list:
			thefile.write("%s\n" % i)

		plt.plot(final_list,'g', label="Vector A")
		plt.title('$ \delta^{-2} \langle x^2(t) \\rangle$ - $\langle x(t) \\rangle^2$ as a function of t')
		plt.xlabel('t')
		plt.ylabel('$\langle x^2(t) \\rangle$ - $\langle x(t) \\rangle^2$')
		plt.legend(loc="upper left")
		plt.show()	
