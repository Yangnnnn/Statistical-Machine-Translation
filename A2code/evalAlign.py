#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle
import math
import array as arr

import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    if use_cached==False:
        return lm_train(data_dir, language, fn_LM)
    if use_cached==True and os.path.exists(fn_LM+".pickle")==True:
        file = open(fn_LM+".pickle","rb")
        return pickle.load(file)
    else:
        return lm_train(data_dir, language, fn_LM)

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    if use_cached==False:
        return align_ibm1(data_dir, num_sent, max_iter, fn_AM)
    if use_cached==True and os.path.exists(fn_AM+".pickle")==True:
        file = open(fn_AM+".pickle","rb")
        return pickle.load(file)
    else:
        return align_ibm1(data_dir, num_sent, max_iter, fn_AM)

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    bleu_array=[]
    
    for i in range(len(eng_decoded)):
        candidate = eng_decoded[i]
        reference_ha = eng[i]
        reference_google = google_refs[i]
        google_len = len(reference_google.split())
        ha_len = len(reference_ha.split())
        can_len = len(candidate.split())
        if abs(google_len-can_len) < abs(ha_len-can_len):
            brevity = google_len/can_len
        else:
            brevity = ha_len/can_len
        if brevity < 1:
            BP=1
        else:
            BP = math.exp(1 - brevity)
        
        if n == 1:
            p1 = BLEU_score(candidate, [reference_ha,reference_google], 1, brevity=False)
            bleu = BP * (p1)**(1/n)
            
        if n == 2:
            p1 = BLEU_score(candidate, [reference_ha,reference_google], 1, brevity=False)
            p2 = BLEU_score(candidate, [reference_ha,reference_google], 2, brevity=False)
            bleu = BP * (p1*p2)**(1/n)
        if n == 3:
            p1 = BLEU_score(candidate, [reference_ha,reference_google], 1, brevity=False)
            p2 = BLEU_score(candidate, [reference_ha,reference_google], 2, brevity=False)    
            p3 = BLEU_score(candidate, [reference_ha,reference_google], 3, brevity=False)
            bleu = BP * (p1*p2*p3)**(1/n)
        bleu_array.append(bleu)
    return bleu_array
        
        
                      
            
        
def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    

    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    
    f = open("Task5.txt", 'w+')
    f.write(discussion) 
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")
    data_dir = "/Users/chulong/Desktop/csc401a2/A2_SMT/data/Hansard/Training"
    
    LM = _getLM(data_dir, "e", "LM", use_cached=True)
    AM_1000 = _getAM(data_dir, 1000, 10, "AM_1000", use_cached=True)
    AM_10000 = _getAM(data_dir,10000, 10, "AM_10000", use_cached=True)
    AM_15000 = _getAM(data_dir,15000, 10, "AM_15000", use_cached=True)
    AM_30000 = _getAM(data_dir,30000, 10, "AM_30000", use_cached=True)
    
    AMs = [AM_1000,AM_10000,AM_15000,AM_30000]
    AM_names = ["AM_1000","AM_10000","AM_15000","AM_30000"]
    french = []
    
    eng = []
    google_refs = []
    
    with open("/Users/chulong/Desktop/csc401a2/A2_SMT/data/Hansard/Testing/Task5.f") as myfile:
        firstNlines = myfile.readlines()[0:25]
    with open("/Users/chulong/Desktop/csc401a2/A2_SMT/data/Hansard/Testing/Task5.google.e") as myfile:
        firstNlines_2 = myfile.readlines()[0:25]
    with open("/Users/chulong/Desktop/csc401a2/A2_SMT/data/Hansard/Testing/Task5.e") as myfile:
        firstNlines_3 = myfile.readlines()[0:25]
        
    for count in firstNlines_2:
    
        google_refs.append(preprocess(count,"e"))
        
    for count in firstNlines_3:
                
        eng.append(preprocess(count,"e"))
        
    for count in firstNlines:
        
        french.append(preprocess(count,"f"))
    
    for i, AM in enumerate(AMs):
        eng_decoded = []
        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        # Decode using AM #
        task5= "/Users/chulong/Desktop/csc401a2/A2_SMT/data/Hansard/Testing/Task5.f"
        
        for count in french:
            eng_decoded.append(decode.decode(count, LM, AMs[i]))
                
        # Eval using 3 N-gram models #
        all_evals = []
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(eng_decoded, eng, google_refs, n)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)