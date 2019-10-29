# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:02:38 2019

@author: stany
"""
import pandas as pd
import numpy as np 
import operator
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


##Import the spambase dataset and adjust as necessary 
spambase = pd.read_csv('spambase.data',header=None)
spambase.rename(columns={0:"word_freq_make", 1:"word_freq_address", 2:"word_freq_all", 3:"word_freq_3d", 4:"word_freq_our", 
                    5:"word_freq_over", 6:"word_freq_remove", 7:"word_freq_internet", 8:"word_freq_order", 9:"word_freq_mail",
                    10:"word_freq_receive", 11:"word_freq_will", 12:"word_freq_people", 13:"word_freq_report", 14:"word_freq_addresses",
                    15:"word_freq_free", 16:"word_freq_business", 17:"word_freq_email", 18:"word_freq_you", 19:"word_freq_credit", 
                    20:"word_freq_your", 21:"word_freq_font", 22:"word_freq_000", 23:"word_freq_money", 24:"word_freq_hp", 
                    25:"word_freq_hpl", 26:"word_freq_george", 27:"word_freq_650", 28:"word_freq_lab", 29:"word_freq_labs", 
                    30:"word_freq_telnet", 31:"word_freq_857", 32:"word_freq_data", 33:"word_freq_415", 34:"word_freq_85", 
                    35:"word_freq_technology", 36:"word_freq_1999", 37:"word_freq_parts", 38:"word_freq_pm", 39:"word_freq_direct", 
                    40:"word_freq_cs", 41:"word_freq_meeting", 42:"word_freq_original", 43:"word_freq_project", 44:"word_freq_re",
                    45:"word_freq_edu", 46:"word_freq_table", 47:"word_freq_conference", 48:"char_freq_;", 49:"char_freq_(", 
                    50:"char_freq_[", 51:"char_freq_!", 52:"char_freq_$", 53:"char_freq_#", 54:"capital_run_length_average", 
                    55:"capital_run_length_longest", 56:"capital_run_length_total", 57:"is_spam"},inplace=True)
#inplace: Makes changes in original Data Frame if True.


##Preprocess data into usable dataframes
sb_spam = spambase[spambase['is_spam']==1].sample(n=78)
sb_ham = spambase[spambase['is_spam']==0].sample(n=122)
sb_200 = pd.concat([sb_spam, sb_ham ], ignore_index=True).reset_index()
sb_200 = sb_200.drop(['index'],axis=1)

sbknn_features = sb_200.drop(['is_spam'],axis=1)
sbknn_response = sb_200['is_spam']
sbknn_response = sbknn_response.reshape((200,1))

sbknn_features = preprocessing.StandardScaler().fit_transform(sbknn_features)
sbknn_response = np.array(sbknn_response).reshape(200,1)

xTrn_knn, xTst_knn, yTrn_knn, yTst_knn = train_test_split(sbknn_features,sbknn_response, test_size = 0.5, random_state = 123, stratify=sbknn_response)

class knn: 
    ''' 
    Classifies data based on Euclidean Distance from training points.  
    '''

    def __init__(self, feature_training_set, feature_test_set, response_training_set, response_test_set, k):  
        self.feature_training_set = feature_training_set
        self.feature_test_set = feature_test_set
        self.response_training_set = response_training_set
        self.response_test_set = response_test_set
        self.k = k
        
    def euclidean_distance(self,a,b):
        '''
        Calculates Euclidean Distance for kNN.
        Takes the X Training Set and the X Testing Set, and finds the Euclidean Distance.  
        '''
#         a = self.feature_training_set 
#         b = self.feature_test_set 
        
        return(np.sqrt(np.sum(np.square(a-b),axis=1)))

    def classify(self):
        '''
        Classifies data using distance calculations and by employing a majority vote amongst k-nearest-neighbors.  
        '''
        
        Xtrain = self.feature_training_set
        Xtest = self.feature_test_set
        Ytrain = self.response_training_set
        Ytest = self.response_test_set
        k = self.k 

        #Assure that ytrain assumes correct dimensions
        m,n = np.shape(Xtest) #m should be 100
        Ytrain = np.array(Ytrain.reshape(m,1)) #Should be 100x1 dimensions

        predictions = [] 

        #Call euclidean distance fxn to compute ed 
        for i in range(m):
            votes = {}
            #we need to get the distances between each test row against every
            #train row all at once as a matrix
            distances = self.euclidean_distance(Xtrain,Xtest[i,:])
            distances = distances.reshape((m,1)) 

            #we need to concatenate Xtrain, Ytest, and distance together 
            training_df = np.concatenate((Xtrain,Ytrain),axis=1) 
            complete_df = np.concatenate((training_df, distances),axis=1)

            #sort array in ascending order (i.e., so that smallest distances come first)
            #but we need to sort by the last column, 
            #whilst keeping everything in the same row tied together (argsort)
            sorted_df = complete_df[complete_df[:,-1].argsort()]

            #return k nearest neighbors (labels are at index -2) 
            knn_df = sorted_df[range(k),-2] 

            #tallies votes; adds an entry to votes dict for new labels 
            #and automatically gives it a vote.  Increments by one if it 
            #sees it again.  
            for i in knn_df: 
                if i not in votes:
                    votes[i] = 1
                else:
                    votes[i] += 1

            #finds the max value, and then retrieves the associated key 
            #this key is our predicted classification (1,0 in this case)  
            pred = max(votes.items(),key=operator.itemgetter(1))[0]

            predictions.append(pred) 

        predictions = np.array(predictions)
        print("Accuracy Score: ", metrics.accuracy_score(Ytest, predictions))
        print("Error Score: ", 1-metrics.accuracy_score(Ytest, predictions))
        return(predictions)
        
knn = knn(xTrn_knn, xTst_knn, yTrn_knn, yTst_knn, 10)
knn.classify()
