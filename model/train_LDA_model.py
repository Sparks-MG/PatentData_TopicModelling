##load dependencies

#imports related to storage retrieval

import sys
import pandas as pd
import numpy as np
import joblib
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:,.2f}'.format #set print float option

## Local Path!
localPath = r'C:\Users\marti\Dropbox\Promotion\Fortbildung\Udacity\Nanodegree\Projekt4\TopicModelling\Code-Bausteine'


#imports related to NLP
import re
import nltk
nltk.download(['punkt', 'stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

#imports related to ML and pipeline

from sklearn.feature_extraction.text import CountVectorizer # TfidfTransformer not needed?
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline # usage of pipeline avoids data leakage
from sklearn.model_selection import GridSearchCV

## functions
def get_csv_data(csv_filepath):
    """
    Loads the data from the csv file defined by Introduction.py
    input: .string representing the full path to the csv file
    output: pandas dataframe without the NaN entries.
    """
    df = pd.read_csv(csv_filepath)
    #split in x,y and something else for test/Training split
    ###See how many values are missing
    print('The percentages of missing values are: \n{}.'.format(df.isnull().mean()*100))
    print('We drop only the rows/patents with NaN for full-text entries.')
    df = df.dropna(subset=['full-text'], axis='rows') # drop only NaN of missing full-text

    #umwandeln der timestamps

    return df


def tokenize(text): #spaeter noch einzel Worter dictionary entfernen?
    '''tokenizes raw text messages
    input:
    output:

    '''

    # normalize case, remove punctuation, special characters and numbers
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r"[^a-zA-Z]+", " ", text.lower())

    # remove single use words is incorporated in count vectorizer
    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove (english) stop words using list comprehension
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    cleanTokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return cleanTokens



def build_lda_model():
    '''Builds the non-pipeline version of the  ML model for the tokenized patent data.
    input: None
    output: Python object representing a grid search model
    '''

    # Define Search Param:  # Number of topics # learning rate
    search_parameters = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
    # Init the Model
    lda_model = LatentDirichletAllocation() #parameter tweeken? paper tricks?
    # Init Grid Search Object
    model = GridSearchCV(lda_model, param_grid=search_parameters)

    return model # execute non-pipeline version fit with model.fit

def build_lda_pipeline_model(): ## Not working yet, dummy for future versions!
    '''Builds the pipeline version of the  ML model for the tokenized patent data. Using the pipeline minimizes data leakage which occurs, when the grid search (automatically) performs cross-fold validation
    input: None
    output: Python object representing the grid search model.
    '''
    # pipeline employs custom function 'tokenize' as tokenizer and initialized CountVectorizer
    pipeline = Pipeline([
    ('vect', CountVectorizer(max_df = 0.8, min_df = 2, stop_words = 'english', tokenizer=tokenize)),
    ('lda', LatentDirichletAllocation())
    ])


    # Define Search Param
    # pipeline version -> estimator.get_params().keys()
    #search_parameters = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
    #'lda__learning_offset  max_df???
    search_parameters = {'lda__n_components': [5, 10, 15, 20, 25, 30],'lda__learning_decay': [.5, .7, .9]}

    # Init Grid Search Object
    model = GridSearchCV(pipeline, param_grid=search_parameters,verbose=10)

    return model # execute pipeline version fit with model().fit

def get_optimizedModelParameter(lda_model,doc_term_matrix):
    '''gets the paramteres of the optimized model'''
    # Best Model
    best_lda_model = lda_model.best_estimator_
    # Model Parameters
    print("Best Model's Params: ", lda_model.best_params_)
    # Log Likelihood Score
    print("Best Log Likelihood Score: ", lda_model.best_score_)
    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(doc_term_matrix))


def save_model(model, model_filepath):
    """
    Saves the trained model to a pickle file at provided path under highly efficient compression (reduces original file size to 1/5).
    input: Python object representing trained classification model, path for saving
    output: (None)
    """
    #compress 3 has been found to be a good compromise between speed and filesize
    joblib.dump(model, model_filepath,compress=3)

def get_topicTopWords(trainedLDAModel, count_vect):
    '''gets the top <20> words from each of the topics
    input: trained model, count_vector instance
    output: list with the <20> top words
    '''
    topic_TopTokens = []

    for topic in trainedLDAModel.components_:
        topic_TopTokens.append(topic.argsort()[-20:])# 20 top tokens per topic
    topTopicWords = []
    for topic in topic_TopTokens:
        dummy = []
        for i in topic:
            dummy.append(count_vect.get_feature_names()[i])
        topTopicWords.append(dummy)
    return topTopicWords

def print_TopicTopicWordes(topTopicWords):
    '''simpy prints number of top topic words to the terminal
    topTopicWords Dimensions: Number of Topics x Number of Top Words
    '''
    topicCounter=0
    for topic in topTopicWords:
        print('Topic {} contains the words:'.format(topicCounter))
        print(topic)
        topicCounter +=1



## execute code

## Prepare data for model
print('Preparing data...')
csv_filepath =localPath + r'\datasets\\patentFullTextData.csv'
patent_df = get_csv_data(csv_filepath)
##Splitting the data according to sections

#get number of patent sections
numPatentSections = patent_df['section'].nunique()
patentSection = patent_df['section'].value_counts().index


patentSectionList = []
for i in range(numPatentSections):
    #generate data frame for each patent section and put it in a list
    patentSectionList.append(patent_df.loc[patent_df['section'] == patentSection[i]])


#patent_df_A = patent_df.loc[patent_df['section'] == 'A']



## initialize instance of CounterVectorizer class for each section
count_vectList = []
#count_vect = CountVectorizer(max_df=0.8, min_df=2)
for i in range(numPatentSections):
    count_vectList.append(CountVectorizer(max_df=0.8, min_df=2))


## Apply fit_transform method of instance on the dataset to generate the vocabulary for each section
print('Generating doc term matrix for each patent section...')

doc_termList = []
for i in range(numPatentSections):
    # generate doc term matrix for each patent section
    doc_termList.append(count_vectList[i].fit_transform(patentSectionList[i]['full-text'].values.astype('U')))
    print('doc term matrix for section {} is generated!'.format(patentSection[i]))


#doc_term_matrix_A = count_vect.fit_transform(patent_df_A['full-text'].values.astype('U'))

print('doc term matrix generated for each patent section...')



## build model for each patent section
print('Building a model for each patent section...')

modelList = []
for i in range(numPatentSections):
    # generate lda model for each patent section
    modelList.append(build_lda_model())

#model_A = build_lda_model() # None Pipeline Version
## for future use: model_A = build_lda_pipeline_model()


## train gridsearch model  for each patent section
print('Performing grids search on the model of each patent section...')

for i in range(numPatentSections):
    # train lda model for each patent section
    modelList[i].fit(doc_termList[i])
    print('Model for section {} is trained!'.format(patentSection[i]))


#model_A.fit(doc_term_matrix_A)   # grid search!


## Get the parameters / information on the optimised LDA model - optional for debugging use
#get_optimizedModelParameter(model_A,doc_term_matrix_A)


## saving the best model out of the grid search object

#model_A_filepath = localPath + r'\model\\lda_model_A.pkl'

modelFilePathList = []
for i in range(numPatentSections):
    modelFilePathList.append(localPath + r'\model\\lda_model_' + patentSection[i] + '.pkl')
    print('Saving model for  patent section {} to filepath:\n: {}'.format(patentSection[i], modelFilePathList[i]))
    save_model(modelList[i].best_estimator_, modelFilePathList[i])

print('Trained model for each patent section saved!')

#print('Saving model for  patent section A to...\n    MODEL: {}'.format(model_A_filepath))
#save_model(model_A.best_estimator_, model_A_filepath)


## Loading pretrained (best model) model again:
# load the pre trained model
#model_A = joblib.load(model_A_filepath)

modelBackupList = []
for i in range(numPatentSections):
    modelBackupList.append(joblib.load(modelFilePathList[i]))


## Displaying some Stuff -count vector fuer alle modelle gleich?


allTopicWords = []
for i in range(numPatentSections):
    topTopicWords = get_topicTopWords(modelBackupList[i], count_vectList[i])
    allTopicWords.append(topTopicWords)


#print("Topics found via LDA:")
#print_TopicTopicWordes(topTopicWords)



print('Saving all Top Words to Single File...')
topTopicWords_filepath = localPath + r'\model\\alltopWords.pkl'
joblib.dump(allTopicWords, topTopicWords_filepath)


##This can be used to classify the patents to the newly found topics (after slight modication)  - add a column to the original data frame that will store the topic for the text

#topic_Probvalues_A = model_A.transform(doc_term_matrix_A)

# transforms data according to fitted model - has 5 columns where each column corresponds to the unormalized probability value of a particular topic
topicProbvaluesList = []
for i in range(numPatentSections):
    topicProbvaluesList.append(modelBackupList[i].transform(doc_termList[i]))


# find the topic index with maximum value - argmax() method and pass 1 as the value for the axis parameter
#patent_df_A['Topic'] = topic_Probvalues_A.argmax(axis=1)


for i in range(numPatentSections):
    patentSectionList[i]['Topic'] = topicProbvaluesList[i].argmax(axis=1)

# patent_df_A.loc[patent_df_A['Topic']==4]

## Save newly topic-labelled data for later data visualization
#patent_df_A.to_csv(localPath+'\datasets\\cleanPatentData_SectionA.csv', index = False)

for i in range(numPatentSections):
    patentSectionList[i].to_csv(localPath+'\datasets\\cleanPatentData_Section_' + patentSection[i] + '.csv', index = False)

print('Classified patent data via trained LDA Model!')