import re

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function
    # if language is english
    start = "SENTSTART "
    end = " SENTEND"
    out_sentence = in_sentence.strip().lower()
    
    if language == "e":
        out_sentence = re.sub(r'([,:;()\-+<>=.?!*/"])',r' \1 ',out_sentence)

        
    if language == "f":
        out_sentence = re.sub(r'([,:;()\-+<>=.?!*/"])',r' \1 ',out_sentence)

        #for l', I think this we do not have to do this step since next step covers this
        out_sentence = re.sub(r'(\b)(l\')(\w+)',r'\1\2 \3',out_sentence)
        #for consonant assume y is not a consonant
        out_sentence = re.sub(r'(\b)([aeiouqwrtypsdfghjklzxcvbnm]\')(\w+)',r'\1\2 \3',out_sentence)
        #for que
        out_sentence = re.sub(r'(\b)(qu\')(\w+)',r'\1\2 \3',out_sentence)
        #for on and il
        out_sentence = re.sub(r'(\w+)(\')(on|il)(\b)',r'\1\2 \3\4',out_sentence)
        #for d’abord, d’accord, d’ailleurs, d’habitude special cases
        out_sentence = re.sub(r'(d\') (abord|accord|ailleurs|habitude)(\b)',r'\1\2\3',out_sentence)
        
    out_sentence = start + out_sentence + end
    out_sentence = re.sub(r' {2,}',r' ',out_sentence)    
    return out_sentence