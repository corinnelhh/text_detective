import csv
from csv import excel_tab
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.cross_validation import cross_val_score
from bs4 import BeautifulSoup
import cPickle
import numpy as np

# 'Blog author gender classification data set associated with the paper
# (Mukherjee and Liu, EMNLP-2010)'
# from http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html


class Detective_(object):
    def __init__(self, gender_data='texts/blog-gender-dataset.csv'):
        self.gender_data = gender_data
        self.clf = self.load_pickle('classifier')
        self.vocab = self.load_pickle('vocab')

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
        clf = MNB(alpha=1E-2)
        print "Running cross-validation."
        score = np.mean(cross_val_score(clf, X, y, cv=10))
        print score
        return clf.fit(X, y)

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
        clf = self.fit_classifier(X, Y)
        print "Finishing fitting classifier"
        return (clf, 'classifier'), (vocab, 'vocab')

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

    def prettify_prediction(self, sample, pred, prob, top_fts):
        genders = {"M": 'man', "F": 'woman'}
        snip = self.get_snippet(sample).encode('utf-8')
        with open('texts/prediction.txt', 'r') as f:
            p = f.read()
        prediction = p.format(
            snip, str("%.2f" % prob)[2:], genders[pred], top_fts
        )
        return prediction

    def show_most_informative_features(self, n=20):
        u"""Code adapted from stack overflow discussion;
        http://stackoverflow.com/questions/11116697/
        how-to-get-most-informative-features-for-scikit-learn-classifiers"""
        coefs_with_fns = sorted(zip(self.clf.coef_[0], self.vocab))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

    def show_features_from_sample(self, sample, pred, n=10):
        coefs_with_fns = sorted(zip(self.clf.coef_[0], self.vocab))
        top = zip(coefs_with_fns, coefs_with_fns[::-1])
        out, sample_w = [], sample.split()
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            w = fn_2 if pred[0] == 'M' else fn_1
            if w in sample_w:
                out.append(w)
            if len(out) > n:
                break
        return ", ".join(out)

    def get_snippet(self, sample):
        bits = BeautifulSoup(sample).get_text().split()
        first, last = " ".join(bits[:15]), " ".join(bits[-15:])
        return " [ . . . ] ".join([first, last])

    def test_teller(self, sample):
        test_x, vocab_ = self.vectorize([sample], self.vocab)
        pred = self.clf.predict(test_x)
        prob = max(self.clf.predict_proba(test_x)[0])
        top_fts = self.show_features_from_sample(sample, pred)
        print zip(self.clf.classes_, self.clf.predict_proba(test_x)[0])
        return self.prettify_prediction(sample, pred[0], prob, top_fts)

if __name__ == '__main__':
    ft = Detective_()
    ft.pickle_prediction_tools()
    # ft.train_teller()
    # ft.show_most_informative_features()
