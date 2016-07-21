import numpy as np
import random

class GeneticAlgorithm(object):
	
	def create_population():

		pop = []
		
		for i in xrange(0,5):
			tmp = [random.randint(0,1) for j in xrange(0,5)]
			pop.append(tmp)

		return pop

	def single_crossover(parent_1, parent_2):
		
		parent_1 = ''.join(str(e) for e in parent_1)
		parent_2 = ''.join(str(e) for e in parent_2)

		cross_over_1_1 = parent_1[:3]
		cross_over_1_2 = parent_2[:3]

		child1 = parent_1.replace(parent_1[:3], cross_over_1_2)
		child2 = parent_2.replace(parent_2[:3], cross_over_1_1)

		return child1, child2

	def three_parent_crossover(parent_1, parent_2, parent_3):

		print "Parent 1:", parent_1
		print "Parent 2:", parent_2
		print "Parent 3:", parent_3
		
		child = []

		for i,j,k in zip(parent_1,parent_2,parent_3):
		    if i == j:
		   		child.append(i)
		    else:
		        child.append(k)

		child = ''.join(child)
		print "Child:", child

		return child

	def random_mutation(generation):
		
		index = random.randrange(len(generation))
	
		gene_to_mutate = generation[index]

		list_gene_to_mutate = list(gene_to_mutate)

		index = random.randint(0,len(list_gene_to_mutate)-1)
		
		for i, j in enumerate(list_gene_to_mutate):
			if i == index:
				if j == "0":
					list_gene_to_mutate[i] = 1
				elif j == "1":
					list_gene_to_mutate[i] = 0
				else:
					raise Exception("Chromosome is not supported!")
		
		mutated_gen = ''.join(str(e) for e in list_gene_to_mutate)
		generation[index] = mutated_gen
		
		return generation

	if __name__ == '__main__':
		
		pop = create_population()
		new_gen = []
		three_new_gen = []

		
		for i in xrange(0,len(pop)-1):
			c1, c2 = single_crossover(pop[i], pop[i+1])
			new_gen.append(c1)
			new_gen.append(c2)

		print "Single Crossover Generation", new_gen
		mutated_gen = random_mutation(new_gen)
		
		
		for i in xrange(0, len(pop)-2):
			child = three_parent_crossover(pop[i], pop[i+1], pop[i+2])
			three_new_gen.append(child)
			
		print "3 Crossover Generation:", three_new_gen
		mutated_gen = random_mutation(new_gen)
		