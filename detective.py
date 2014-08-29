import csv
from csv import excel_tab
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import cross_val_score
import numpy as np

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
        return lr, vocab

    def test_teller(self, sample):
        lr, vocab = self.train_teller()
        print "teller trained"
        test_x, vocab_ = self.vectorize([sample], vocab)
        print len(test_x)
        print "vectorized sample"
        prediction = lr.predict(test_x)
        return prediction[0]

if __name__ == '__main__':
    with open('julia.txt', 'r') as f:
        m = f.read()
    ft = FortuneTeller()
    print ft.test_teller(m)
