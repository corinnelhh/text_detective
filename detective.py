import csv
from csv import excel_tab
from sklearn.feature_extraction.text import CountVectorizer as CV
import numpy as np

# 'Blog author gender classification data set associated with the paper
# (Mukherjee and Liu, EMNLP-2010)'
# from http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html


class FortuneTeller(object):
    def __init__(self, gender_data='texts/blog-gender-dataset.csv'):
        self.gender_data = gender_data

    def vectorize(self, X, y):
        stopwords = open('texts/stopwords.txt').read().lower().split()
        vec = CV(
            analyzer='word',
            stop_words=stopwords,
            encoding='latin-1',
        )
        print "Building X, Y..."
        X = vec.fit_transform(X).toarray()
        Y = np.array(y)
        return X, Y, vec.get_feature_names()

    def read_gender_data_file(self):
        lines = csv.reader(open(self.gender_data, 'rU'), dialect=excel_tab)
        X, y = [], []
        print "Reading in files"
        for line in lines:
            line = [i for i in line[0].split(',') if len(i)]
            if len(line):
                y.append(line.pop())
                X.append(" ".join(line))
        print "Read in files"
        return X, y

    def train_teller(self):
        X, Y = self.read_gender_data_file()
        X, Y, vocab = self.vectorize(X, Y)

if __name__ == '__main__':
    ft = FortuneTeller()
    ft.train_teller()
