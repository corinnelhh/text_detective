import csv
from csv import excel_tab
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import cross_val_score
import cPickle
import numpy as np
import random

# 'Blog author gender classification data set associated with the paper
# (Mukherjee and Liu, EMNLP-2010)'
# from http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html


class Detective_(object):
    def __init__(self, gender_data='texts/blog-gender-dataset.csv'):
        self.gender_data = gender_data

    def vectorize(self, X, vocab_=None):
        stopwords = open('texts/stopwords.txt').read().lower().split()
        vec = TFIDF(
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
        print "Running cross-validation."
        score = np.mean(cross_val_score(lr, X, y, cv=10))
        print score
        return lr.fit(X, y)

    def read_gender_data_file(self):
        lines = csv.reader(open(self.gender_data, 'rU'), dialect=excel_tab)
        X, y = [], []
        labels = ['M', 'F']
        print "Reading in files"
        for line in lines:
            line = [i for i in line[0].split(',') if len(i)]
            if len(line):
                g = line.pop().strip().upper()
                if g in labels:
                    y.append(g)
                    X.append(" ".join(line))
        print "Read in files"
        return X, y

    def train_teller(self):
        X, y = self.read_gender_data_file()
        Y = np.array(y)
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
        pickle_file.close()
        print item, 'pickle loaded'
        return X

    def prettify_prediction(self, pred):
        responses = {'M': "male_preds.txt", 'F': "female_preds.txt"}
        with open('texts/%s' % responses[pred], 'r') as f:
            responses = f.readlines()
        return random.choice(responses)

    def show_most_informative_features(self, n=20):
        u"""Code adapted from stack overflow discussion;
        http://stackoverflow.com/questions/11116697/
        how-to-get-most-informative-features-for-scikit-learn-classifiers"""
        clf = self.load_pickle('classifier')
        vocab = self.load_pickle('vocab')
        coefs_with_fns = sorted(zip(clf.coef_[0], vocab))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

    def test_teller(self, sample):
        lr = self.load_pickle('classifier')
        vocab = self.load_pickle('vocab')
        test_x, vocab_ = self.vectorize([sample], vocab)
        prediction = lr.predict(test_x)
        print zip(lr.classes_, lr.predict_proba(test_x)[0])
        return self.prettify_prediction(prediction[0])

if __name__ == '__main__':
    ft = Detective_()
    # ft.pickle_prediction_tools()
    ft.train_teller()
    ft.show_most_informative_features()
