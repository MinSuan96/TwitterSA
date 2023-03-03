import glob
import numpy as np
import utils
import pandas as pd
from sklearn.metrics import classification_report

# Takes majority vote of a number of CSV prediction files.

NUM_PREDICTION_ROWS = 21553
TEST_LABEL_FILE = '../twitter_data/bigDataset/Twitter_Data-processed-y-test.csv'
REPORT_FILE = './reports/majority-voting.csv'


def main():
    csvs = ['3cnn.csv', '4cnn.csv', 'cnn-feats-svm-linear-1.00-1000.csv', 'cnn_maxlength_20.csv', 'lstm.csv']
    predictions = np.zeros((NUM_PREDICTION_ROWS, 2))
    for csv in csvs:
        with open(csv, 'r') as f:
            lines = f.readlines()[1:]
            current_preds = np.array([int(l.split(',')[1]) for l in lines])
            predictions[list(range(NUM_PREDICTION_ROWS)), current_preds] += 1
    print(predictions[:50])
    predictions = np.argmax(predictions, axis=1)
    results = list(zip(list(map(str, list(range(NUM_PREDICTION_ROWS)))), predictions))
    utils.save_results_to_csv(results, 'majority-voting.csv')
    test_label = utils.file_number_to_list(TEST_LABEL_FILE)
    report = classification_report(test_label, predictions, output_dict=True)
    print(classification_report(test_label, predictions, output_dict=False))
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(REPORT_FILE)


if __name__ == '__main__':
    main()
