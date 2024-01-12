from sbmr_MBKBOSSVS import *
from pyts.datasets import *
import pandas as pd
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import time

k_list = [32,64]
curr_name = 'SBMR-MB-K-BOSS-VS'

representations_list = ['unigram','bigram']
names_list = ['Crop','FordB','FordA','NonInvasiveFetalECGThorax2','NonInvasiveFetalECGThorax1','PhalangesOutlinesCorrect','HandOutlines','TwoPatterns']

curr_do_chi2 = True
curr_numerosity= False
curr_do_trend = True

window_sizes=[0.1, 0.2,0.3,0.4, 0.5,0.6, 0.7,0.8, 0.9]

window_step_list =[(0.0125)]*len(window_sizes)

for curr_representation in representations_list:
    for curr_k in k_list:
        for curr_dataset in names_list:
            (data_train, data_test, target_train, target_test)=fetch_ucr_dataset(curr_dataset, use_cache=True, data_home='/Users/aglenis/ucr_datasets/',
                                                                                 return_X_y=True)
            try:
                clf = SCALE_BOSS_MR_MBKBOSSVS(window_size_list=window_sizes,window_step_list=window_step_list,
                                  representation=curr_representation,strategy= 'uniform',do_chi2=curr_do_chi2,numerosity_reduction=curr_numerosity,do_trend=curr_do_trend,k=curr_k)
                start_fit = time.time()
                clf.fit(data_train, target_train)
                end_fit = time.time()
                start_score = time.time()
                curr_score = clf.score(data_test, target_test)
                end_score = time.time()
                time_fit = end_fit-start_fit
                time_score = end_score-start_score
                print(curr_name+','+str(curr_k)+','+curr_dataset+','+str(curr_score)+','+str(time_fit)+','+str(time_score)+','+str(time_fit+time_score)+','+str(curr_representation)+','+str(curr_do_chi2)+','+str(window_sizes)+','+str(window_step_list))
            except Exception as e:
                print(traceback.format_exc())
                pass
