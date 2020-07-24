#load dependencies

#imports related to storage retrieval
import sys
import pandas as pd
import numpy as np
import joblib

localPath = r'C:\Users\marti\Dropbox\Promotion\Fortbildung\Udacity\Nanodegree\Projekt4\TopicModelling\Code-Bausteine'
topTopicWords_filepath = localPath + r'\model\\alltopWords.pkl'

#imports related to visualization
import matplotlib.pyplot as plt
import wordcloud


# Get text - the words were previously sorted by probability, so this will match the visualization
completeTextList = joblib.load(topTopicWords_filepath)

##Split the list with the dimension nPatentSections x nTopics x nTopwords:

# completeTextList[0] patentSection A, all Topics, number of Top Words
# completeTextList[0][0] patentSection A, Topic 0, number of Top Words

def saveWordCloud(textList,localPath, imageName):
    '''This is a helper function which simply plots the content of alltopWords.pkl as word clouds '''
    # Convert list to single string
    textAsString=(" ").join(textList)
    # Create and generate a word cloud image:
    cloud_object = wordcloud.WordCloud(background_color="white",include_numbers=True).generate(textAsString)
    plt.figure()
    plt.imshow(cloud_object, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(localPath + r'\wordClouds' +imageName)

# this is correct for the employed data set
patentSectionOrder = ['G','H', 'A','B', 'C','F', 'E', 'D']

# For each patent section
for i in range(len(completeTextList)):
    # For each topic per patent section
    for j in range(len(completeTextList[i])):
        imageName = r'\\' + patentSectionOrder[i] + '_Topic_' + str(j)  +'.png'
        saveWordCloud(completeTextList[i][j], localPath,imageName)
