from sbmr_clf import *
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
from sklearn.utils import check_random_state
from sklearn.linear_model import RidgeClassifierCV

import time
import numpy as np


clf_list = []
clf_names_list = []
clf_list.append(RidgeClassifierCV(alphas=np.logspace(-1, 5, 10)))
clf_names_list.append("SBMR-ridge-cv-W0-trend-D6-UG")



representations_list = ['unigram']

names_list = ['Crop','FordB','FordA','NonInvasiveFetalECGThorax2','NonInvasiveFetalECGThorax1','PhalangesOutlinesCorrect','HandOutlines','TwoPatterns']

curr_do_chi2 = True
curr_numerosity= False
curr_do_trend = True

total_time = 0.0
total_accuracy = 0.0
for curr_representation in representations_list:
    for curr_name, curr_clf in zip(clf_names_list, clf_list):
        for curr_dataset in names_list:
            (data_train, data_test, target_train, target_test)=fetch_ucr_dataset(curr_dataset, use_cache=True, data_home='/Users/aglenis/ucr_datasets/',
                                                                                 return_X_y=True)
            try:

                window_sizes=[24]
                window_step_list = [1]*len(window_sizes)

                clf = SCALE_BOSS_MR(curr_clf,window_size_list=window_sizes,window_step_list=window_step_list,
                                  representation=curr_representation,strategy= 'uniform',do_chi2=curr_do_chi2,numerosity_reduction=curr_numerosity,do_trend=curr_do_trend,
                                  dilation_padding_flag = True,dilation_padding_size =10,dilation_list=[1,9])
                start_fit = time.time()
                clf.fit(data_train, target_train)
                end_fit = time.time()
                start_score = time.time()
                curr_score = clf.score(data_test, target_test)
                end_score = time.time()
                time_fit = end_fit-start_fit
                time_score = end_score-start_score
                print(curr_name+','+curr_dataset+','+str(curr_score)+','+str(time_fit)+','+str(time_score)+','+str(time_fit+time_score)+','+str(curr_representation)+','+str(curr_do_chi2)+','+str(window_sizes)+','+str(window_step_list))
                total_time+= (time_fit+time_score)
                total_accuracy+=curr_score
            except Exception as e:
                print(traceback.format_exc())
                pass
print('Average Accuracy : '+str(total_accuracy/len(names_list)))
print('Average Execution time : '+str(total_time/len(names_list)))
