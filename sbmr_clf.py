"""Code for SCALE-BOSS-MR """

# Author: Apostolos Glenis


import numpy as np
from math import ceil
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from pyts.approximation import SymbolicFourierApproximation
from pyts_base import UnivariateClassifierMixin
from pyts.utils.utils import _windowed_view

from sklearn.feature_selection import chi2
#from dilation_utils import *

from numba import (  # set_num_threads,
    NumbaPendingDeprecationWarning,
    NumbaTypeSafetyWarning,
    njit,
    objmode,
    prange,
)
from numba.core import types
from numba.typed import Dict


def _dilation(X, d, first_difference,do_padding = False,padding_size =10):
    if do_padding:
        padding = np.zeros((len(X), padding_size))
        X = np.concatenate((padding, X, padding), axis=1)

    # using only first order differences
    if first_difference:
        X = np.diff(X, axis=1, prepend=0)

    # adding dilation
    X_dilated = _dilation2(X, d)
    X_index = _dilation2(np.arange(X_dilated.shape[-1], dtype=np.float_)
                         .reshape(1, -1), d)[0]

    return (
        X_dilated,
        X_index,
    )


@njit(cache=True, fastmath=True)
def _dilation2(X, d):
    # dilation on actual data
    if d > 1:
        start = 0
        data = np.zeros(X.shape, dtype=np.float_)
        for i in range(0, d):
            curr = X[:, i::d]
            end = curr.shape[1]
            data[:, start : start + end] = curr
            start += end
        return data
    else:
        return X.astype(np.float_)

class SCALE_BOSS_MR(BaseEstimator, UnivariateClassifierMixin):
    """Bag-of-SFA Symbols in Vector Space.

    Each time series is transformed into an histogram using the
    Bag-of-SFA Symbols (BOSS) algorithm. Then, for each class, the histograms
    are added up and a tf-idf vector is computed. The predicted class for
    a new sample is the class giving the highest cosine similarity between
    its tf vector and the tf-idf vectors of each class.

    Parameters
    ----------
    word_size : int (default = 4)
        Size of each word.

    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and 26.

    window_size : int or float (default = 10)
        Size of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_size * n_timestamps)``.

    window_step : int or float (default = 1)
        Step of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_step * n_timestamps)``.

    anova : bool (default = False)
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    drop_sum : bool (default = False)
        If True, the first Fourier coefficient (i.e. the sum of the subseries)
        is dropped. Otherwise, it is kept.

    norm_mean : bool (default = False)
        If True, center each subseries before scaling.

    norm_std : bool (default = False)
        If True, scale each subseries to unit variance.

    strategy : str (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution
        - 'entropy': Bin edges are computed using information gain

    alphabet : None, 'ordinal' or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used.

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurence of back to back
        identical occurences of the same words.

    use_idf : bool (default = True)
        Enable inverse-document-frequency reweighting.

    smooth_idf : bool (default = False)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool (default = True)
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        An array of class labels known to the classifier.

    idf_ : array, shape = (n_features,) , or None
        The learned idf vector (global term weights) when ``use_idf=True``,
        None otherwise.

    tfidf_ : array, shape = (n_classes, n_words)
        Term-document matrix.

    vocabulary_ : dict
        A mapping of feature indices to terms.

    References
    ----------
    .. [1] P. Schäfer, "Scalable Time Series Classification". Data Mining and
           Knowledge Discovery, 30(5), 1273-1298 (2016).

    Examples
    --------
    >>> from pyts.classification import BOSSVS
    >>> from pyts.datasets import load_gunpoint
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = BOSSVS(window_size=28)
    >>> clf.fit(X_train, y_train)
    BOSSVS(...)
    >>> clf.score(X_test, y_test)
    0.98

    """

    def __init__(self,clf, word_size=4, n_bins=4,  window_size_list=[10], window_step_list=[1],
                 anova=False, drop_sum=False, norm_mean=False, norm_std=False,
                 strategy='uniform', alphabet=None, numerosity_reduction=False,
                 use_idf=False, smooth_idf=False, sublinear_tf=True,
                 representation= 'unigram',do_chi2=False,chi2_threshold=2,do_trend=True,
                 dilation_padding_flag = True,dilation_padding_size =10,dilation_list=[1]):
        self.word_size = word_size
        self.n_bins = n_bins
        self.window_size_list = window_size_list
        self.window_step_list = window_step_list
        self.anova = anova
        self.drop_sum = drop_sum
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.strategy = strategy
        self.alphabet = alphabet
        self.numerosity_reduction = numerosity_reduction
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.clf = clf
        self.representation = representation
        self.do_chi2=do_chi2
        self.chi2_threshold=chi2_threshold
        self.do_trend= do_trend

        self._window_size_list = []
        self._window_step_list = []
        self._n_windows_list = []
        self._tfidf_list = []
        self._sfa_list = []
        self.vocabulary_list = []
        self.tfidf_list = []
        self.skip_gram_tfidf_list = []

        self._window_size_trend_list = []
        self._window_step_trend_list = []
        self._n_windows_trend_list = []
        self._tfidf_trend_list = []
        self._sfa_trend_list = []
        self.vocabulary_trend_list = []
        self.tfidf_trend_list = []
        self.skip_gram_tfidf_trend_list = []

        self.relevant_features_list = []
        self.relevant_features_trend_list = []

        self.dilation_list = dilation_list
        self.dilation_padding_flag = dilation_padding_flag
        self.dilation_padding_size = dilation_padding_size


    def process_time_series(self,X,y):
        X, y = check_X_y(X, y)
        n_samples, n_timestamps = X.shape
        check_classification_targets(y)
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = self.classes_.size

        window_size, window_step = self._check_params(n_timestamps)
        n_windows = (n_timestamps - window_size + window_step) // window_step

        X_windowed = _windowed_view(
            X, n_samples, n_timestamps, window_size, window_step
        )
        X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)

        sfa = SymbolicFourierApproximation(
            n_coefs=self.word_size, drop_sum=self.drop_sum, anova=self.anova,
            norm_mean=self.norm_mean, norm_std=self.norm_std,
            n_bins=self.n_bins, strategy=self.strategy, alphabet=self.alphabet
        )
        y_repeated = np.repeat(y, n_windows)
        X_sfa = sfa.fit_transform(X_windowed, y_repeated)

        X_word = np.asarray([''.join(X_sfa[i])
                             for i in range(n_samples * n_windows)])
        X_word = X_word.reshape(n_samples, n_windows)

        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1],
                              np.full(n_samples, True)]
            X_bow = np.asarray([' '.join(X_word[i, not_equal[i]])
                                for i in range(n_samples)])
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

        #X_class = np.array([' '.join(X_bow[y_ind == i])
        #                    for i in range(n_classes)])
        ngram_param = None
        if self.representation == 'unigram':
            ngram_param = (1,1)
        else:
            ngram_param = (1,2)
        tfidf = TfidfVectorizer(
            norm=None, use_idf=self.use_idf, smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,ngram_range=ngram_param
        )
        train_normal = tfidf.fit_transform(X_bow).toarray()
        if self.representation == 'skipgram':
            import functools
            from nltk.util import skipgrams
            skipper = functools.partial(skipgrams, n=2, k=2)
            skip_gram_tfidf = TfidfVectorizer(
                norm=None, use_idf=self.use_idf, smooth_idf=self.smooth_idf,
                sublinear_tf=self.sublinear_tf,analyzer=skipper )
            transformed_train_skip = skip_gram_tfidf.fit_transform(X_bow).toarray()
            final_train = np.hstack([train_normal,transformed_train_skip])
        else:
            final_train =train_normal
            skip_gram_tfidf = None

        if self.do_chi2 == True:
            chi2_statistics, _ = chi2(final_train, y_ind)
            relevant_features = np.where(
                chi2_statistics > self.chi2_threshold)[0]

            final_train = final_train[:,relevant_features]
            relevant_return = relevant_features
        else:
            relevant_return = [1]*final_train.shape[1]

        ret_vocabulary = {value: key for key, value in
                            tfidf.vocabulary_.items()}

        if self.use_idf:
            self.idf_ = tfidf.idf_
        else:
            self.idf_ = None



        return final_train,ret_vocabulary,window_size,window_step,n_windows,tfidf,sfa,skip_gram_tfidf,relevant_return
    def fit(self, X, y):
        """Compute the document-term matrix.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : array-like, shape = (n_samples,)
            Class labels for each data sample.

        Returns
        -------
        self : object

        """
        n_samples, n_timestamps = X.shape
        to_be_fitted2 = np.array([], dtype=np.float64).reshape(n_samples,0)
        for (curr_window_size_train,curr_window_step_train) in zip(self.window_size_list,self.window_step_list):
            for curr_dilation in self.dilation_list:
                #print(str(curr_window_size_train)+','+str(curr_dilation))
                X_dilated,X_index = _dilation(X=X, d = curr_dilation, first_difference=False,do_padding = self.dilation_padding_flag,padding_size =self.dilation_padding_size)

                self.window_size =curr_window_size_train
                self.window_step =curr_window_step_train

                final_train,ret_vocabulary,window_size_train,window_step_train,n_windows_train,tfidf,sfa,skip_gram_tfidf,relevant_return = self.process_time_series(X_dilated,y)

                self._window_size_list.append(window_size_train)
                self._window_step_list.append(window_step_train)
                self._n_windows_list.append(n_windows_train)
                self._tfidf_list.append(tfidf)
                self._sfa_list.append(sfa)
                self.vocabulary_list.append(ret_vocabulary)
                self.tfidf_list.append(final_train)
                self.skip_gram_tfidf_list.append(skip_gram_tfidf)
                to_be_fitted2 = np.hstack([to_be_fitted2,final_train])
                self.relevant_features_list.append(relevant_return)
        if self.do_trend:
            X_trend = np.diff(X)
            for (curr_window_size_train,curr_window_step_train) in zip(self.window_size_list,self.window_step_list):
                for curr_dilation in self.dilation_list:
                    #print(str(curr_window_size_train)+','+str(curr_dilation))
                    X_dilated_trend,X_index = _dilation(X_trend, d = curr_dilation, first_difference=False,do_padding = self.dilation_padding_flag,padding_size=self.dilation_padding_size)

                    self.window_size =curr_window_size_train
                    self.window_step =curr_window_step_train



                    final_train_trend,ret_vocabulary_trend,window_size_trend,window_step_trend,n_windows_trend,tfidf_trend,sfa_trend,skip_gram_tfidf_trend,relevant_return_trend = self.process_time_series(X_dilated_trend,y)

                    self._window_size_trend_list.append(window_size_trend)
                    self._window_step_trend_list.append(window_step_trend)
                    self._n_windows_trend_list.append(n_windows_trend)
                    self._tfidf_trend_list.append(tfidf_trend)
                    self._sfa_trend_list.append(sfa_trend)
                    self.vocabulary_trend_list.append(ret_vocabulary_trend)
                    self.tfidf_trend_list.append(tfidf_trend)
                    self.skip_gram_tfidf_trend_list.append(skip_gram_tfidf_trend)
                    self.relevant_features_trend_list.append(relevant_return_trend)

                    #to_be_fitted = np.hstack([final_train,final_train_trend])
                    to_be_fitted2 = np.hstack([to_be_fitted2,final_train_trend])


        le = LabelEncoder()
        y_ind = le.fit_transform(y)

        self.clf.fit(to_be_fitted2 ,y_ind)


        return self


    def process_test_time_series(self,X,_window_size,_window_step,_n_windows,_sfa,_tfidf,skip_gram_tfidf,curr_relevant_features):
        #check_is_fitted(self, ['vocabulary_', 'tfidf_', 'idf_', '_tfidf'])
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape

        X_windowed = _windowed_view(
            X, n_samples, n_timestamps,_window_size, _window_step
        )
        X_windowed = X_windowed.reshape(-1,_window_size)

        X_sfa = _sfa.transform(X_windowed)
        X_word = np.asarray([''.join(X_sfa[i]) for i in range(X_sfa.shape[0])])
        X_word = X_word.reshape(n_samples,_n_windows)

        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1],
                              np.full(n_samples, True)]
            X_bow = np.asarray([' '.join(X_word[i, not_equal[i]])
                                for i in range(n_samples)])
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
        X_tf_normal =_tfidf.transform(X_bow).toarray()
        if self.representation == 'skipgram':
            X_tf_skip = skip_gram_tfidf.transform(X_bow).toarray()
            X_tf = np.hstack([X_tf_normal,X_tf_skip])
        else:
            X_tf = X_tf_normal
        if self.idf_ is not None:
            X_tf /= self.idf_

        if self.do_chi2 == True:
            X_tf = X_tf[:,curr_relevant_features]

        return X_tf

    def decision_function(self, X):
        """Evaluate the cosine similarity between document-term matrix and X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        X : array, shape (n_samples, n_classes)
            Cosine similarity between the document-term matrix and X.

        """
        n_samples,n_timestamps = X.shape
        final_test_array = np.array([], dtype=np.float64).reshape(n_samples,0)


        for (curr_window_size,curr_window_step,curr_dilation,curr_n_windows,curr_sfa,curr_vectorizer,curr_skipgram,curr_relevant_features) in zip(self._window_size_list,self._window_step_list,self.dilation_list*len(self.window_size_list),self._n_windows_list,self._sfa_list,self._tfidf_list,self.skip_gram_tfidf_list,self.relevant_features_list):
            #print(str(curr_window_size)+','+str(curr_dilation))
            X_dilated_test,X_index = _dilation(X, d = curr_dilation, first_difference=False,do_padding = self.dilation_padding_flag,padding_size=self.dilation_padding_size)
            X_tf_normal = self.process_test_time_series(X_dilated_test,curr_window_size,curr_window_step,curr_n_windows,curr_sfa,curr_vectorizer,curr_skipgram,curr_relevant_features)
            final_test_array = np.hstack([final_test_array,X_tf_normal])

        if self.do_trend:
            X_trend = np.diff(X)
            for (curr_window_size_trend,curr_window_step_trend,curr_dilation,curr_n_windows_trend,curr_sfa_trend,curr_vectorizer_trend,curr_skipgram_trend,curr_relevant_features_trend) in zip(self._window_size_trend_list,self._window_step_trend_list,self.dilation_list*len(self.window_size_list),self._n_windows_trend_list,
                           self._sfa_trend_list,self._tfidf_trend_list,self.skip_gram_tfidf_trend_list,self.relevant_features_trend_list):
                #print(str(curr_window_size_trend)+','+str(curr_dilation))
                X_dilated_trend,X_index = _dilation(X_trend, d = curr_dilation, first_difference=False,do_padding = self.dilation_padding_flag,padding_size=self.dilation_padding_size)

                X_tf_trend = self.process_test_time_series(X_dilated_trend,curr_window_size_trend,curr_window_step_trend,curr_n_windows_trend,curr_sfa_trend,curr_vectorizer_trend,curr_skipgram_trend,curr_relevant_features_trend)
                #X_tf_trend = X_tf_trend[:,curr_relevant_features]
                final_test_array = np.hstack([final_test_array,X_tf_trend])

        classes_list =[]
        n_classes = self.classes_.size

        ret_array = self.clf.predict(final_test_array)
        return ret_array
        #return cosine_similarity(X_tf, self.tfidf_)

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        y_pred : array, shape = (n_samples,)
            Class labels for each data sample.

        """
        return self.classes_[self.decision_function(X)]

    def _check_params(self, n_timestamps):
        if not isinstance(self.word_size, (int, np.integer)):
            raise TypeError("'word_size' must be an integer.")
        if not self.word_size >= 1:
            raise ValueError("'word_size' must be a positive integer.")

        if not isinstance(self.window_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_size' must be an integer or a float.")
        if isinstance(self.window_size, (int, np.integer)):
            if self.drop_sum:
                if not 1 <= self.window_size <= (n_timestamps - 1):
                    raise ValueError(
                        "If 'window_size' is an integer, it must be greater "
                        "than or equal to 1 and lower than or equal to "
                        "(n_timestamps - 1) if 'drop_sum=True'."
                    )
            else:
                if not 1 <= self.window_size <= n_timestamps:
                    raise ValueError(
                        "If 'window_size' is an integer, it must be greater "
                        "than or equal to 1 and lower than or equal to "
                        "n_timestamps if 'drop_sum=False'."
                    )
            window_size = self.window_size
        else:
            if not 0 < self.window_size <= 1:
                raise ValueError(
                    "If 'window_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1."
                )
            window_size = ceil(self.window_size * n_timestamps)

        if not isinstance(self.window_step,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_step' must be an integer or a float.")
        if isinstance(self.window_step, (int, np.integer)):
            if not 1 <= self.window_step <= n_timestamps:
                raise ValueError(
                    "If 'window_step' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps."
                )
            window_step = self.window_step
        else:
            if not 0 < self.window_step <= 1:
                raise ValueError(
                    "If 'window_step' is a float, it must be greater "
                    "than 0 and lower than or equal to 1."
                )
            window_step = ceil(self.window_step * n_timestamps)

        if self.drop_sum:
            if not self.word_size <= (window_size - 1):
                raise ValueError(
                    "'word_size' must be lower than or equal to "
                    "(window_size - 1) if 'drop_sum=True'."
                )
        else:
            if not self.word_size <= window_size:
                raise ValueError(
                    "'word_size' must be lower than or equal to "
                    "window_size if 'drop_sum=False'."
                )
        return window_size, window_step
