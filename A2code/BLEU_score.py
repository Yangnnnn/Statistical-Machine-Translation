import math

def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on
    
    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.
    
    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
	
	#TODO: Implement by student.
    

    C = 0
    candidate_tokens = candidate.split()
    N = 0	
    refs = []
    if n == 1:
        N = len(candidate_tokens)
        for ref in references:
            refs += ref.split()
    
        for token in candidate_tokens:
            if token in refs:
                C = C+1
        bleu_score = C/N

    if n == 2:
        N = len(candidate_tokens)-1
        for ref in references:
            ref_tokens = ref.split()
            for ref_idx in range(len(ref_tokens)):
                if ref_idx+1 < len(ref_tokens):
                    refs.append([ref_tokens[ref_idx],ref_tokens[ref_idx+1]])
        for can_idx in range(len(candidate_tokens)):
            if can_idx+1 < len(candidate_tokens):
                if [candidate_tokens[can_idx],candidate_tokens[can_idx+1]] in refs:
                    C = C+1
        bleu_score = C/N

    if n == 3:
        N = len(candidate_tokens)-2
        for ref in references:
            ref_tokens = ref.split()
            for ref_idx in range(len(ref_tokens)):
                if ref_idx+2 < len(ref_tokens):
                    refs.append([ref_tokens[ref_idx],ref_tokens[ref_idx+1],ref_tokens[ref_idx+2]])
        for can_idx in range(len(candidate_tokens)):
            if can_idx+2 < len(candidate_tokens):
                if [candidate_tokens[can_idx],candidate_tokens[can_idx+1],candidate_tokens[can_idx+2]] in refs:
                    C = C+1
        bleu_score = C/N
	    
	    
	    
    if brevity == True:
        references_len = []
        candidate_len = len(candidate_tokens)
        for ref in references:
            references_len.append(len(ref.split()))
	
        for i in range(len(references_len)):
            if i == 0:
                min_idx = 0
                min_len = abs(references_len[i]-candidate_len)
            if abs(references_len[i]-candidate_len) < min_len:
                min_len = abs(references_len[i]-candidate_len)
                min_idx = i
        min_len = references_len[min_idx]
        brevity = min_len/candidate_len
	
        if brevity < 1:
            BP = 1
        if brevity >= 1:
            BP = math.exp(1 - brevity)
        bleu_score = bleu_score*BP
	    
        
        
	
    return bleu_score