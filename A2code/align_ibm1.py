from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    
    # Read training data
    data = read_hansard(train_dir, num_sentences)
    eng = data[0]
    fre = data[1]
    # Initialize AM uniformly
    AM = initialize(eng, fre)
    
    # Iterate between E and M steps
    for i in range(max_iter):
        print(i)
        em_step(AM, eng, fre)
	
    with open(fn_AM +'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)   

    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	output: return (eng, fre)
	"""
    count = 0
    finished = []
    files_names = sorted(os.listdir(train_dir))
    english_files = []
    french_files = []
    eng = []
    fre = []

    for name in files_names:
        name1 = name.split(".")
        if name1[-1] == "e" :
            if name[:-1]+"f" in files_names and name[:-1]+"e" in files_names:
                english_files.append(os.path.join(train_dir,name))
                french_files.append(os.path.join(train_dir,name[:-1]+"f"))
		
    
    
    for i in range(len(english_files)):
        file_e = open(english_files[i],"r").read().splitlines()
        file_f = open(french_files[i],"r").read().splitlines()
	
        for j in range(len(file_e)):
            if len(eng) < num_sentences:
                if file_e[j].split() != [] and file_f[j].split() != []:
                    eng.append(preprocess(file_e[j],"e").split())
                    fre.append(preprocess(file_f[j],"f").split())
	    
            else:
                return (eng,fre)

    return (eng,fre)
	    

def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	# TODO
    AM = {}
    eng_token = {}
    for tokens in range(len(eng)):
        for token in eng[tokens]:
            if token != "SENTSTART" and token != "SENTEND":
                if token not in eng_token:
                    eng_token[token] = list(set(fre[tokens][1:-1]))
                else:
                    eng_token[token] = list(set(eng_token[token]).union(set(fre[tokens][1:-1])))
    
    for e in eng_token:
        for f in eng_token[e]:
            if e not in AM:
                AM[e]={}
                AM[e][f] = 1/len(eng_token[e])
            else:
                AM[e][f] = 1/len(eng_token[e])
    
    AM['SENTSTART'] = {}
    AM['SENTEND'] = {}   
    AM['SENTSTART']['SENTSTART'] = 1
    AM['SENTEND']['SENTEND'] = 1
    return AM
		

    
	
	
    
def em_step(t, eng, fre):
    """
    One step in the EM algorithm.
    Follows the pseudo-code given in the tutorial slides.
    """
    # TODO
    tcount = {}
    total = {}
    # need to modify
    for e in t:
        tcount[e] = {}
        total[e] = 0
        for f in t[e]:
            tcount[e][f] = 0
    del tcount["SENTEND"]
    del tcount["SENTSTART"]
    del total["SENTEND"]
    del total["SENTSTART"]
	
    for i in range(len(eng)):
        eng_sen = eng[i][1:-1]
        fre_sen = fre[i][1:-1]
        unique_f = list(set(fre_sen))
        unique_e = list(set(eng_sen))
        for f in unique_f:
            denom_c = 0
            for e in unique_e:
                denom_c += t[e][f] * fre_sen.count(f)	
            for e in unique_e:
                tcount[e][f] += t[e][f] * fre_sen.count(f) * eng_sen.count(e) / denom_c
                total[e] += t[e][f] * fre_sen.count(f) * eng_sen.count(e) / denom_c
    for e in total:
        for f in tcount[e]:
            t[e][f] = tcount[e][f] / total[e] 
    return t
	