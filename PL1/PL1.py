# Name: Abhishek Patel
# Couse: Data Mining CSE-5334-005
# Student ID: 1002033618
#Assigment: Programming Assigment P1

# importing basic libs
import math
import os
import time
import copy
import warnings
warnings.filterwarnings("ignore")
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    os.system("python3 -m pip install tqdm")
    from tqdm import tqdm
try:
    import pandas as pd
except ModuleNotFoundError:
    print("Python 3 module (Pandas) not found. Downloading Pandas...")
    os.system("python3 -m pip install pandas")
    import pandas as pd


# DataFrame to store doc data and tfidf data
D = pd.DataFrame(columns=["Doc_No","File_Name","Raw_Doc","Tokenized_Doc","Doc_Without_Stopwords","Doc_After_Stemming"])
T = pd.DataFrame(columns = ["Term", "Term_Freq", "Term_InvDoc_Freq","TFIDF"])
idf = {}
tf = {}
NO_OF_DOCS = 30


# results1 Function to get the idf value of a given token. args: token.
# we fetch the value stored in the data frame.
# if it exists we print the value if it does not exists we print -1
def getidf(token):
    try:
        res = T.loc[T.Term == token, 'Term_InvDoc_Freq'].values[0]
    except IndexError:
        res = float(-1)
    print("%.12f"%res)


# results2 Function to get the weight of a token for a particualr file. args: filename and token.
# we fetch the value stored in the data frame.
# if it exists we print the value if it does not exists we print 0
def getweight(filename,token):
    try:
        required = D.loc[D.File_Name == filename, 'Doc_No'].values[0]
        weight = T.loc[T.Term == token,'TFIDF'].values[0][required]
    except IndexError:
        weight = 0
    print("%.12f"%weight)

#results3 Function to calculate the cosine similarity between query and document. args: doc vectoer, query vector and doc details.
# we just multiply the doc vectoer with query vector and sum up the final vlaues of each term.
def cosine_similarity(doc_weight, query_weight, doc_deatils):
    if doc_deatils['flag'] == 1:
        res = {key : doc_weight[key] * query_weight[key] for key in query_weight}
        res = sum (res.values())
        return res
    if doc_deatils['flag'] == 0:
        return ["None",0]

# Task 5 Function for Calculating Term Frequency. args: document tokens after stemming.  
# Using for loop to count the number of items a particular word has occured in the documnet.
# Using dictonary to easy up the usage later on.
def term_frequency(doc):
    temp_tf = {}
    for item in doc:
        if item in temp_tf: 
            temp_tf[item] +=1
        else:
            temp_tf[item] = 1
    for key in temp_tf:
        temp_tf[key] = (1+math.log10(temp_tf[key])) ## formual for term frequency
    return temp_tf

# Task 4 Function for performing stemming. args: document tokens after stopword removal.
# Using PorterStemmer for stemming. Inbuild function of nltk pakage.

def stemming(doc):
    try:
        from nltk.stem.porter import PorterStemmer
    except:
        os.system('python3 -m pip install nltk')
        from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    res = list(map(lambda x : stemmer.stem(x), doc)) ## using map function to iterate over all the tokens
    return res

# Task 3 Function for stopword removal. args: tokens after tokanizing document.
# Using nltk stopwords package to identify list of stop words and remove them from the list tokens 
def stopword_removal(tokens):
    try:
        from nltk.corpus import stopwords
        res = list(filter(lambda x: x not in stopwords.words('english'), tokens))
    except LookupError:
        print("nltk.stopwords module not found Downloaduing nltk.stopwords")
        import nltk
        nltk.download('stopwords')
        res = list(filter(lambda x: x not in stopwords.words('english'), tokens))
    return res

# Task 2 Function for tokanization of documents. args: raw document in lower case. 
def tokenization(doc):
    doc = doc.replace("'","").replace("-","")
    from nltk.tokenize import RegexpTokenizer
    tokenReg = RegexpTokenizer(r"[a-zA-Z0-9]+")
    tokens = tokenReg.tokenize(doc)
    return tokens

# function to analysis query and calculate the cosine similarity and final output
def qurey(string):
    #First starting with preprocessing the query string.
    query = string.lower()
    query_tokenize = tokenization(query)
    query_stop_words_removal = stopword_removal(query_tokenize)
    query_stemming = stemming(query_stop_words_removal)
    ## calculating the term frequency for the query.
    query_weight = term_frequency(query_stemming)
    posting_list = {}         #Initializing the posting list  

    # Fetching all the tfidf values of documents having terms matching with query string tokens.
    # need try except block to prevent error if the term does not exists in the data frame.
    for keys in query_weight:
        try:
            tfidf = T.loc[T.Term == keys,'TFIDF'].values[0]
            tfidf = sorted(tfidf.items(),  key=lambda x:x[1], reverse=True)
            tfidf = dict(list(tfidf)[0: 10])
        except IndexError:
            tfidf = []
        posting_list[keys] = tfidf   #Adding TFIDF value dictonary in postlign list
    # if all the list in posting list are empty meaning non of the terms in query are present in document.
    # As a result just endinf the program here itself.
    res = all(len(x) == 0 for x in posting_list.values())
    if res:
        output = cosine_similarity({}, {}, {"file_name": "None", "flag": 0})
        return output
    # if more than 0 list in posting list are empty calculating cosine similary
    else:
        cossim = {key:0 for key in range(0,30)}
        for doc_count in range(0,30):
            doc_deatils = {"file_name": D.loc[D.Doc_No == i,'File_Name'].values[0],"flag":1}
            doc_weight = {key:0 for key in query_weight}
            # using a for loop to calculate the document vector.
            for items in posting_list:
                if doc_count in posting_list[items].keys():
                    doc_weight[items] = posting_list[items][doc_count]
                else:
                    doc_weight[items] = list(posting_list[items].values())[-1]
            cossim[doc_count] = cosine_similarity(doc_weight, query_weight, doc_deatils)
            
        # Sorting the calculated cosin similarity values to find the largestone.
        cossim = sorted(cossim.items(),  key=lambda x:x[1], reverse = True)
        cossim = dict(list(cossim))
        
        output = [D.loc[D.Doc_No == list(cossim.keys())[0],'File_Name'].values[0],list(cossim.values())[0]]
        # just checking thr fech more condition here. 
        count = 0
        for items in list(posting_list.values()):
            if list(cossim.keys())[0] in list(items.keys()):
                count += 1
        if count > 1:
            return output
        else:
            return ["Fetch more", 0]
        

# Task 1 Reading the files and storing them in dataframe. This is the main driver function.
if __name__ == "__main__":
    start = time.time()
    print("Reading Data Files....")
    corpusroot = './presidential_debates'   # location of files i.e the folder containg the txt file should be in the same folder as this program.
    count = 0
    # Running a loop on the folder to import the files. 
    for filename in tqdm(os.listdir(corpusroot)):
        count +=1
        file = open(os.path.join(corpusroot,filename), encoding='UTF-8')
        doc = file.read()
        doc_in_lower_case = doc.lower()
        # pandas provide easy access and editting of data so using pandas dataframe to store the data.
        D = D.append({'Doc_No':count-1,"File_Name":filename,"Raw_Doc":doc_in_lower_case,"Tokenized_Doc":[],"Doc_Without_Stopwords":[],"Doc_After_Stemming":[]},ignore_index=True)
    print("All Files Imported.")


    # Here we start our subproblems each function definded above is called one after another in subsequent order.
    print("Perfomring pre-processing.....")
    temp_list = []
    for idx,row in D.iterrows(): # just for better appreances on terminal nothing else.
        temp_list +=[row]
    # for each document calling the indivuals subproblem function.
    for row in tqdm(temp_list):
        doc_in_lower_case = row['Raw_Doc']
        doc_after_tokenization = tokenization(doc_in_lower_case)
        row['Tokenized_Doc'] = doc_after_tokenization
        doc_after_stopword_removal = stopword_removal(doc_after_tokenization)
        row['Doc_Without_Stopwords'] = doc_after_stopword_removal
        doc_after_stemming = stemming(doc_after_stopword_removal)
        row['Doc_After_Stemming'] = doc_after_stemming
    print("Pre-Processing Completed.")


    # Calculation for term frequency starts here.
    print("Calculating TF values.....")
    temp_list = []
    for idx,row in D.iterrows():     # just for better appreances on terminal nothing else.
        temp_list +=[row]
    docs_done = 0

    for row in tqdm(temp_list):
        term_freq = term_frequency(row['Doc_After_Stemming'])
        # As the term may or maynot  appear in ever document need to make correction in term frequency list for each term for each document. 
        # So if a term does not appreas in first 2 document we add 0 in thoes values and append the term frequency of third document.
        for key in term_freq:
            if key in tf:
                for i in range(len(tf[key]),docs_done):
                    tf[key][i]=0
                tf[key][docs_done]=term_freq[key]
            else:
                tf[key] = {}
                for i in range(docs_done):
                    tf[key][i]=0
                tf[key][docs_done]=term_freq[key]
        # keeping track of document frequency so we dont have to redo this loop again.
            if key in idf:
                idf[key] +=1
            else:
                idf[key] = 1
        docs_done += 1
    # adding correction values for last documents if the term never appeared after 5th document we add 0 to all the documents after 5 for this term.
    for keys in tf:
        for i in range(len(tf[keys]),NO_OF_DOCS):
                    tf[keys][i] = 0
    print("TF values Calculation Completed.")


    # Calculating TF-IDF values.
    print("Calculating TF-IDF....")
    mag_tfidf = {key:0 for key in range(0,30)}  # dictionary to keep track of magnitude of tfidf to normalize it later.

    # for all the terms we keep track of document frequecy before so just using the formula to claculate idf vlaues.
    for keys in tqdm(idf):
        tfidf = {}
        idf_values = math.log10(NO_OF_DOCS/idf[keys])   # formula to claculate idf vlaues.
        tf_values = tf[keys]
        tfidf = copy.deepcopy(tf_values)
        # for all the different term frequency of different document calculatign the tf-idf values. 
        for key in tfidf:
            tfidv_val = tfidf[key] * idf_values
            tfidf[key] = tfidv_val
            # updating the magnitude of tf-idf vector. 
            mag_tfidf[key] = mag_tfidf[key] +(tfidv_val)**2
        new_row = pd.Series([keys,tf_values,idf_values,tfidf], index = T.columns)
        T = T.append(new_row,ignore_index=True)   # Saving the values in dataframe.
    print("TF-IDF Calculation Completed.")

    print("Normalizing TF-IDF values ....")
    # normalizing the tf-idf values by dividing it by the magnitude i.e between 0 and 1.
    for keys in mag_tfidf:
        mag_tfidf[keys] = mag_tfidf[keys] ** 0.5  # taking square root to finalize the value of magnitude.
    temp_list = []
    for idx,row in T.iterrows():     # just for better appreances on terminal nothing else.
        temp_list +=[row]
    for row in tqdm(temp_list):
        nor_tfidf = {key:0 for key in range(0,30)}
    # updating the tf-idf values to normalized values.
        for i in range(0,30):
            res = row["TFIDF"][i]/mag_tfidf[i]
            nor_tfidf[i] = res
        T.iat[idx,3] = nor_tfidf
    print("Normalization of TF-IDF values Completed.")
    

    ###can use the below lines to see the data frame by storing them in excel files. 
    # try:
    #     D.to_excel(r'PreProcessedData.xlsx', index = False)
    #     T.to_excel(r'TFIDF.xlsx', index = False)
    # except ModuleNotFoundError:
    #     os.system("python3 -m pip install openpyxl")
    #     D.to_excel(r'PreProcessedData.xlsx', index = False)
    #     T.to_excel(r'TFIDF.xlsx', index = False)

    print("\n #################  OUTPUT ################# \n")
    getidf("health")
    getidf("agenda")
    getidf("vector")
    getidf("reason")
    getidf("hispan")
    getidf("hispanic")
    getweight("2012-10-03.txt","health")
    getweight("1960-10-21.txt","reason")
    getweight("1976-10-22.txt","agenda")
    getweight("2012-10-16.txt","hispan")
    getweight("2012-10-16.txt","hispanic")
    print(qurey("health insurance wall street"))
    print(qurey("particular constitutional amendment"))
    print(qurey("terror attack"))
    print(qurey("vector entropy"))
    end = time.time()
    print("\n Totle time required for execution:",end-start," secs.\n")