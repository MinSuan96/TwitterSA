from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from numpy import loadtxt
import numpy as np
import pickle
import utils
import pandas as pd
from sklearn.metrics import classification_report

# Performs SVM classification on features extracted from penultimate layer of CNN model.


TRAIN_FEATURES_FILE = './models/train-feats.npy'
TRAIN_LABELS_FILE = './models/train-labels.txt'
TEST_FEATURES_FILE = './models/test-feats.npy'
TEST_LABEL_FILE = '../twitter_data/3-sentiment-processed-y-test.csv'
REPORT_FILE = './reports/cnn_feats_svm_3sentiments.csv'
CLASSIFIER = 'SVM'
MODEL_FILE = 'cnn-feats-%s.pkl' % CLASSIFIER
TRAIN = False
C = 1
MAX_ITER = 1000

if TRAIN:
    X_train = np.load(TRAIN_FEATURES_FILE)
    y_train = loadtxt(TRAIN_LABELS_FILE, dtype=float).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    if CLASSIFIER == 'SVM':
        model = svm.LinearSVC(C=C, verbose=1, max_iter=MAX_ITER)
        model.fit(X_train, y_train)

    print(model)
    del X_train
    del y_train
    with open(MODEL_FILE, 'wb') as mf:
        pickle.dump(model, mf)
    val_preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_preds)
    print("Val Accuracy: %.2f%%" % (accuracy * 100.0))

else:
    with open(MODEL_FILE, 'rb') as mf:
        model = pickle.load(mf)
    X_test = np.load(TEST_FEATURES_FILE)
    print(X_test.shape)
    test_preds = model.predict(X_test)
    results = zip(map(str, range(X_test.shape[0])), test_preds)
    utils.save_results_to_csv(results, 'cnn-feats-svm-linear-%.2f-%d.csv' % (C, MAX_ITER))
    test_label = utils.file_number_to_list(TEST_LABEL_FILE)
    report = classification_report(test_label, test_preds, output_dict=True)
    print(classification_report(test_label, test_preds, output_dict=False))
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(REPORT_FILE)
