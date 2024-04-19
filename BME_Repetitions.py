##Counts repetition##

# - This code looks at an INPUT dataframe constituted of turns from 3 different speakers, 
# each being displayed in a different column according to the BME method (sample available).
# - The output of this code is an OUTPUT dataframe similar to the input's but augmented with 
# several columns including the number of items available and repeated for each turn, along with the final Jaccard Index.

# The end goal of this code is:
# - From one turn to another, it counts repetitions as a Jaccard Index (i.e. how similar are two turns?)
# It accounts for several types of repetitions: Self-Repetitions, Other-Repetitions.
# It accounts for three natures of repeated items: Open-Class Items, Closed-Class Items, Both undistinctively (also call "all").
# It does the calculation for several n-grams.

## To test this code, a sample of the Multisimo Corpus (first 20 seconds of every conversation is provided) 

# Requirements: 
# Python 3.10.2
# spaCy 3.7.2
# pandas 1.5.0
# numpy 1.23.3

#### 

#imports#
import pandas as pd # for dataframes
import spacy # for the POS tags
 
#load the small English model which has shown to be enough.
nlp = spacy.load("en_core_web_sm")
from spacy.tokens import Doc #to create a Doc object (personalised tokenizer)
 

#Functions#

def my_tokenizer(string): # Specific Tokeniser so it reproduces older paper and deals with the specificities of our notations for spoken dialogue (as opposed to text).

    #normalises the data:
    string = string.lower()  #lowercase
        
    string = string.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").replace("'", " ").replace("[", " ").replace("]", " ") 
    list_of_tokens = string.split() #parses

    try :
        list_of_tokens.remove(["[", "+", "]", ",", "(" , ")", ".", "?", "!", "...", "'", '"']) #removes punctuation
    except(ValueError): 
        print("no punctuation in turn:", list_of_tokens)

    doc = Doc(nlp.vocab, words=list_of_tokens) # makes it a DOC object, usable as Tokeniser for spaCy.
    return doc

nlp.tokenizer = my_tokenizer # Makes it the default tokeniser now.


# Our repetition system compares the last turns of a person to the present turns. The "register" keeps in memory the last turn of every speaker as sets of tokens.
# adds_to_register() adds a given turn to the register under the adequate speaker's ID.
def adds_to_register(ID, list, register):
    # should be ngrams
    new_list = [tuple(item) for item in list] #from the list of paired POS and tokens, keeps the tokens and their given POS as tuples (because set(list) can't work if list is made of lists)
    register[ID] = set(new_list)  # stores a set of the turn, so no word is repeated.
    
    return(ID, register)
    

def counts_rep(ID, string, register, ngram): # Function that counts the repetitions.
    
    # initialises the counts of all ("") nature of tokens, Open class tokens (OC) and Close class tokens (CC)
    count_repeated = {"self" : {"repeated": {"": 0,
                                            "OC" : 0,
                                            "CC" : 0},
                                "nonrepeated": {"": 0,
                                            "OC" : 0,
                                            "CC" : 0},
                                "jaccard_index": {"": 0,
                                            "OC" : 0,
                                            "CC" : 0},
                                "repetition":{"": [],
                                            "OC" : [],
                                            "CC" : []}
                                },
                    "other" : {"repeated": {"": 0,
                                            "OC" : 0,
                                            "CC" : 0},
                                "nonrepeated": {"": 0,
                                            "OC" : 0,
                                            "CC" : 0},
                                "jaccard_index": {"": 0,
                                            "OC" : 0,
                                            "CC" : 0},
                                "repetition":{"": [],
                                            "OC" : [],
                                            "CC" : []}
                        }
                    }

    
    #Tokenises and adds POS to the turns, and "OPEN/CLOSE" class categories
    sentences = string.replace("!", ".").replace("?", ".").split(".") # splits all turns in sentences. 
    list_of_tokens = []

    for i, sent in enumerate(nlp.pipe(sentences)): #for every sentence
        if sent.has_annotation('DEP'): #if has dependencies
            #adds the tokens present in the sentence to the token list
            for word in sent: #for every word
                if word.text == "laugh": #all laughters: forced classification to INTJ
                    word.pos_ = "INTJ"
                elif (word.text[:1] == "mh") or (word.text[:1] == "hm") or (word.text[:1] == "uhm"): #all hesitation: forced classification to INTJ
                    word.pos_ = "INTJ" 
                word_category = "closed" # By default, the category of word is closed 
                if word.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]: #But "open" in these cases
                    word_category = "open"
                
                list_of_tokens.append((word.text, word.pos_, word_category)) #rebuilds the sentence as a list of token; 1 token = (word, pos, category)

    # list of tokens is parsed into n-grams that belong to different categories (all, Open Class, Close Class):
    list_of_ngrams = {"": [],
                      "OC" : [],
                      "CC" : []
                    }
    
    # Divides the list into n-grams
    for i in range(0,(len(list_of_tokens) - (ngram-1))): #for every n-gram
        list_of_ngrams[""].append(tuple(list_of_tokens[i:i+ngram])) 
        
        # Checks whether the n-gram is open (if there is an element that is open, the ngram is open)
        ngram_C = "CC" # the n-gram is by default closed

        for token in (list_of_tokens[i:i+ngram]):
            try:
                if token[2] == "open": #if one of the token is Open
                    ngram_C = "OC" #n-gram is changed to "open"
                    
            except(IndexError):
                quit(f"problem with token: {token}")
            
            list_of_ngrams[ngram_C].append(tuple(list_of_tokens[i:i+ngram]))


    #count repetitions#
    #set items from the register available for repetition: 
            #in the register are arranged by players. Now need to be arranged by "self" (speaker's) or "other"'s content
    set_items = {"other": {"" : [],
                           "OC" : [],
                           "CC": []},
                "self": {"" : [],
                           "OC" : [],
                           "CC": []} 
                }
   
    for participant in register:
        
        for type in set(register[participant]): 

            isOpen = "CC" #tracks if the type, as a n-gram, is open class 
            for subtype in type:
                if subtype[2] == "open":
                    isOpen = "OC"
                if subtype[2] not in ["open", "closed"]:
                    quit(("problem with subtype:", subtype, "subtype[2]:", subtype[2]))
                
            if ID == participant: #if ID of the speaker is the same as the register's participant's, then adds self repetitions

                set_items["self"][""].append(type) #adds to the "all" main category
                set_items["self"][isOpen].append(type) #adds to either the "CC" or "OC" category, based on previous for-loop.
                
            else :  #if the register's participant is not the speaker:
                set_items["other"][""].append(type)
                set_items["other"][isOpen].append(type)

    for C in ["", "OC", "CC"] :  #For three categories: all, OC, CC
        for p in ["self", "other"]: # for the self and other repetitions

            # counts the repetitions 
            for ngram in list_of_ngrams[C]: # for n-gram in turn.

                if ngram in set_items[p][C]: # if n-gram is present in the register for self/other, all/OC/CC
                    count_repeated[p]["repeated"][C] += 1 # count of repetition incremented
                    count_repeated[p]["repetition"][C].append(ngram) # keeping track of what has counted as a repetition

            # keeps track of the words from the register which could have been repeated but were not.
            for type in set(set_items[p][C]):
                if type not in count_repeated[p]["repetition"][C]: 
                        count_repeated[p]["nonrepeated"][C] += 1
                
            # calculates the jaccard index
            try: 
                count_repeated[p]["jaccard_index"][C] = count_repeated[p]["repeated"][C] / (len(set(set_items[p][C])) + len(list_of_ngrams[C]) - count_repeated[p]["repeated"][C])
            except(ZeroDivisionError): #if there is a turn but no n-gram for n > 1
                count_repeated[p]["jaccard_index"][C] = 0

    return(list_of_ngrams, count_repeated)









####
#open files#

#input file
df = pd.read_csv("input_example.csv")

for ngram in range (1,4): #loop add different n-gram sized repetitions
 
    #initialise a dictionary containing all data for every line, before being put into a dataframe to save it.
    dict_final = {"MOD":  {"self" : {"repeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "nonrepeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "jaccard_index": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "length":{"": [],
                                                  "OC" : [],
                                                    "CC" : []},
                                        "repetition": { "": [],
                                                  "OC" : [],
                                                    "CC" : []}},
                            "other" : {"repeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "nonrepeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "jaccard_index": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "length":{"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "repetition": { "": [],
                                                  "OC" : [],
                                                    "CC" : []}}} ,

                    "P1":  {"self" : {"repeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "nonrepeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "jaccard_index": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "length":{"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "repetition": { "": [],
                                                  "OC" : [],
                                                    "CC" : []}},
                            "other" : {"repeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "nonrepeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "jaccard_index": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "length":{"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "repetition": { "": [],
                                                  "OC" : [],
                                                    "CC" : []}}},

                    "P2":  {"self" : {"repeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "nonrepeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "jaccard_index": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "length":{"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "repetition": { "": [],
                                                  "OC" : [],
                                                    "CC" : []}},
                            "other" : {"repeated": {"": [],
                                            "OC" : [],
                                            "CC" : []},
                                        "nonrepeated": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "jaccard_index": {"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "length":{"": [],
                                                    "OC" : [],
                                                    "CC" : []},
                                        "repetition": { "": [],
                                                  "OC" : [],
                                                    "CC" : []}}}
                    }
    
    
    
       
    conversation = "" # nature of the conversation (we span through 18 conversations, so this is important to check the boundaries and
                     # reinitialise it on time.)
   

    

    for l in df.index: # for every line/turn.

        # dict_turn_counts to keep track of the information required on one line (before being added to dict_final once we've processed it all)
        dict_turn_counts = {"MOD":  {"self" : {"repeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "nonrepeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "jaccard_index": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "length":{"": "",
                                                  "OC" : "",
                                                    "CC" : ""},
                                        "repetition":{"": "",
                                                  "OC" : "",
                                                    "CC" : ""}},
                            "other" : {"repeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "nonrepeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "jaccard_index": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "length":{"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "repetition":{"": "",
                                                  "OC" : "",
                                                    "CC" : ""}}} ,

                    "P1":  {"self" : {"repeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "nonrepeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "jaccard_index": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "length":{"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "repetition": { "": "",
                                                  "OC" : "",
                                                    "CC" : ""}},
                            "other" : {"repeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "nonrepeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "jaccard_index": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "length":{"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "repetition": { "": "",
                                                  "OC" : "",
                                                    "CC" : ""}}},

                    "P2":  {"self" : {"repeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "nonrepeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "jaccard_index": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "length":{"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "repetition":{"": "",
                                                  "OC" : "",
                                                    "CC" : ""}},
                            "other" : {"repeated": {"": "",
                                            "OC" : "",
                                            "CC" : ""},
                                        "nonrepeated": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "jaccard_index": {"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "length":{"": "",
                                                    "OC" : "",
                                                    "CC" : ""},
                                        "repetition":{"": "",
                                                  "OC" : "",
                                                    "CC" : ""}}}
                    }

        if (df["Conv_MOD_P1_P2"][l] != conversation): #if new conversation, reinitialises the register, so there is no overlap of repetition.
            register = {    
                        "player1": [],
                        "player2": [],
                        "MOD": []
                    }    
            conversation = df["Conv_MOD_P1_P2"][l] #takes the new conversation_ID

        turns_to_add = {} #turns to add to the register.
   
        for ID in ["MOD", "P1", "P2"]:
            list_of_ngrams = {"":[],
                              "OC":[],
                              "CC":[]}

            if df["BME_Turn_"+ID][l] in ["B_W","B_M"]: # if is a b_turn (i.e. see the BME method), then it means there is a new turn to look into
                string =  str(df["Tag_Turn_"+ID][l])
                list_of_ngrams, dict_turn_counts[ID] = counts_rep(ID, string, register, ngram)
                turns_to_add[ID] = list_of_ngrams[""] #keeps track of the speaker's ID, and their turn.

            #adds necessary turns to the register without changing if a player hasn't uttered anything since last time.
            for participant in turns_to_add:
                string =  turns_to_add[participant]
                adds_to_register(participant, string, register) 

            # transfers to "dict_final" all the information collected and calculated on this line.
            for p in ["self","other"]:
                for C in ["", "OC", "CC"]:
                    for rep in ["repeated", "nonrepeated"]:
                        dict_final[ID][p][rep][C].append(dict_turn_counts[ID][p][rep][C])
                    dict_final[ID][p]["length"][C].append(len(list_of_ngrams[C]))
                    dict_final[ID][p]["jaccard_index"][C].append(dict_turn_counts[ID][p]["jaccard_index"][C])
                    dict_final[ID][p]["repetition"][C].append(dict_turn_counts[ID][p]["repetition"][C])


    # put in the DataFrame    
    for ID in ["MOD", "P1", "P2"]:
        for C in ["", "OC", "CC"]:
            for p in ["self","other"]:
                for rep in ["repeated", "nonrepeated", "length", "jaccard_index", "repetition"]:
                    df[ID+"_"+C+"_"+p+"_"+rep+"_"+str(ngram)+"n"] = dict_final[ID][p][rep][C]
                    

                    

#SaveFile :

## Feels up M and E lines based on the B_lines (see the BME method).  
for ngram in range(1,4):
    for ID in ["MOD", "P1", "P2"]:
        for p in ["self","other"]:
            for rep in ["repeated", "nonrepeated", "length", "jaccard_index"]:
                for C in ["", "OC", "CC"]:
                    for line in df.index:
                        if line != 0: #if line isn't the first one (M and E lines can't be first.)
                            if df['BME_Turn_' + ID][line] in ["M", "E_M", "E_W"]:
                                df[ID+"_"+C+"_"+p+"_"+rep+"_"+str(ngram)+"n"][line] = df[ID+"_"+C+"_"+p+"_"+rep+"_"+str(ngram)+"n"][line-1]

name_output = "output_example.csv"

df.to_csv(name_output)
print(f"Output {name_output} has been created.")
