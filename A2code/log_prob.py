from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
    prob = 1
    tokens = sentence.split()
    if smoothing == False:
        for i in range(len(tokens)):
            current = tokens[i]
            if current != "SENTEND":
                next_word = tokens[i+1]
                # repeat
                if current in LM['bi'] and LM['uni'][current] != 0:
                    if next_word in LM['bi'][current] and LM['bi'][current][next_word] != 0:
                        prob = prob * (LM['bi'][current][next_word])/(LM['uni'][current])
                    else: # next_word not in 
                        prob = 0
                else:
                    prob = 0

    if smoothing == True:
        for i in range(len(tokens)):
            current = tokens[i]
            if current != "SENTEND":
                next_word = tokens[i+1]
                # repeat
                if current in LM['bi'] :
                    count_w1 = LM['uni'][current]
                    if next_word in LM['bi'][current]:
                        count_w = LM['bi'][current][next_word]
                    else:
                        count_w = 0                    
                else:
                    count_w1 = 0
                    count_w = 0

                prob = prob * (count_w+delta)/(count_w1+delta*vocabSize)
                    
                      
                    
    if prob > 0 :
        log_prob = log(prob,2)
    else:
        log_prob = float('-inf')
		    
                   
	
	
            
    return log_prob