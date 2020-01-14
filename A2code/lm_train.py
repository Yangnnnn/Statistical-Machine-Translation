from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
    # TODO: Implement Function
    language_model = {}
    uni = {}
    bi ={}
    #get files
    files = os.listdir(data_dir)
    for i in files:
        temp = i.split(".")
        if temp[-1] == language:
            file_dir = os.path.join(data_dir,i)
            with open(file_dir) as file:
                for line in file:
                    pre_line = preprocess(line, language)
                    tokens = pre_line.split()
                    for idx in range(len(tokens)):
                        word = tokens[idx]
                        if idx+1 < len(tokens):
                            word_next = tokens[idx+1]
                        else:
                            word_next = ""
                            
                        if word not in uni:
                            uni[word] = 1
                        else:
                            uni[word] = uni[word]+1
                        if word_next != "":
                            if word not in bi:
                                bi[word]={}
                                bi[word][word_next] = 1
			
                            else:
                                if word_next not in bi[word]:
                                    bi[word][word_next] = 1
                                else:
                                    bi[word][word_next] = bi[word][word_next] + 1

    language_model["uni"] = uni
    language_model["bi"] = bi
    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return language_model