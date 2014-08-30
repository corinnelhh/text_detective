import csv
from csv import excel_tab
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import cross_val_score
import cPickle
import numpy as np
import random

# 'Blog author gender classification data set associated with the paper
# (Mukherjee and Liu, EMNLP-2010)'
# from http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html


class FortuneTeller(object):
    def __init__(self, gender_data='texts/blog-gender-dataset.csv'):
        self.gender_data = gender_data

    def vectorize(self, X, vocab_=None):
        stopwords = open('texts/stopwords.txt').read().lower().split()
        vec = CV(
            analyzer='word',
            stop_words=stopwords,
            encoding='latin-1',
            vocabulary=vocab_
        )
        print "Building X, Y..."
        X = vec.fit_transform(X).toarray()
        return X, vec.get_feature_names()

    def fit_classifier(self, X, y):
        lr = LR()
        # print "Running cross-validation."
        # score = np.mean(cross_val_score(lr, X, y, cv=10))
        # print score
        return lr.fit(X, y)

    def read_gender_data_file(self):
        lines = csv.reader(open(self.gender_data, 'rU'), dialect=excel_tab)
        X, y = [], []
        print "Reading in files"
        for line in lines:
            line = [i for i in line[0].split(',') if len(i)]
            if len(line):
                g = line.pop().strip().upper()
                if g == 'M':
                    y.append(g)
                    X.append(" ".join(line))
                elif g == 'F':
                    y.append(g)
                    X.append(" ".join(line))
        print "Read in files"
        Y = np.array(y)
        return X, Y

    def train_teller(self):
        X, Y = self.read_gender_data_file()
        X, vocab = self.vectorize(X)
        lr = self.fit_classifier(X, Y)
        print "Finishing fitting classifier"
        return (lr, 'classifier'), (vocab, 'vocab')

    def pickle_prediction_tools(self):
        for el in self.train_teller():
            pickle_file = open('pickles/%s' % el[1], 'wb')
            cPickle.dump(el[0], pickle_file)
            pickle_file.close()
            print "Finished pickling", el[1]

    def load_pickle(self, item):
        pickle_file = open('pickles/%s' % str(item), 'rb')
        X = cPickle.load(pickle_file)
        print "Pickle loaded."
        pickle_file.close()
        return X

    def prettify_prediction(self, prediction):
        responses = {'M': [], 'F': []}
        with open('texts/male_preds.txt', 'r') as f:
            m_responses = f.read()
        with open('texts/female_preds.txt', 'r') as f:
            f_responses = f.read()
        for m_resp in m_responses:
            responses['M'].append(m_resp)
        for f_resp in f_responses:
            responses['F'].append(f_resp)
        return responses

    def test_teller(self, sample):
        lr = self.load_pickle('classifier')
        vocab = self.load_pickle('vocab')
        test_x, vocab_ = self.vectorize([sample], vocab)
        print len(test_x)
        print "vectorized sample"
        prediction = lr.predict(test_x)
        return prediction[0]

if __name__ == '__main__':
    ft = FortuneTeller()
    ft.pickle_prediction_tools()

