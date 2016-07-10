"A Class that reads the content of a csv file and computes the n_grams out of it with nltk. Once this is done the output is filtered out according to the content of a particular file in order to keep only the relevant n_grams"

import csv
import re
import json
from nltk.util import ngrams

class VacumCleanerReviews():
	
	def compute_n_grams(review): #Function that computes n_grams...
						 #change n according to the number of n_gram you want

		review = ''.join(review)

		n = 6
		n_gram = ngrams(review.split(), n)

		with open('TotalN_grams.txt', 'w') as outfile:
			for gram in n_gram:
				json_nGram = json.dumps(gram)
				json.dump(json_nGram, outfile)


	def filter_n_grams(complains):	#A new file is created which only keeps the relevant n_grams, which means only the n_grams that deal with the list of complains

		for complain in complains:
			open('Filtered_Data.txt','w').writelines([ line for line in open('TotalN_grams.txt') if complain in line])

	def extract_complains(): #Function that extracts all the complains of yout .txt file
					 #everything is saved into the vector = single_complains

		file = open('cw2.txt', 'r')
		text = file.read().lower()
		file.close()
		text = re.sub('[^a-z\ \']+', " ", text)
		single_complains = list(text.split())

		return single_complains

	def single_data(): #Function that reads all reviews and extracts
			   # all the words and saves them into vector = review_words

		file = open('data.csv', 'r')
		text = file.read().lower()
		file.close()
		text = re.sub('[^a-z\ \']+', " ", text)
		review_words = list(text.split())

		return review_words

	if __name__ == '__main__':

		complains = extract_complains()
		words_of_review = single_data()

		if [i for i in complains if i in words_of_review]:	#If a word inside the complains list is also present in the review the n_gram is computed
			compute_n_grams(' '.join(words_of_review))

	filter_n_grams(complains)
