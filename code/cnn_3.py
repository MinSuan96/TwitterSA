import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.metrics import classification_report

# Performs classification using CNN.

FREQ_DIST_FILE = '../twitter_data/3-sentiment-processed-train-freqdist.pkl'
BI_FREQ_DIST_FILE = '../twitter_data/3-sentiment-processed-train-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = '../twitter_data/3-sentiment-processed-train.csv'
TEST_PROCESSED_FILE = '../twitter_data/3-sentiment-processed-X-test.csv'
TEST_LABEL_FILE = '../twitter_data/3-sentiment-processed-y-test.csv'
GLOVE_FILE = '../dataset/glove-seeds.txt'
MODEL_FILE = './models/4cnn-08-0.077-0.195.hdf5'
REPORT_FILE = './reports/3-sentiments.csv'
train = False
dim = 200


def get_glove_vectors(vocab):
    """
    Extracts glove vectors from seed file only for words present in vocab.
    """
    print('Looking for GLOVE seeds')
    glove_vectors = {}
    found = 0
    with open(GLOVE_FILE, 'r', encoding='utf-8') as glove_file:
        for i, line in enumerate(glove_file):
            utils.write_status(i + 1, 0)
            tokens = line.strip().split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                glove_vectors[word] = np.array(vector)
                found += 1
    print('\n')
    return glove_vectors


def get_feature_vector(tweet):
    """
    Generates a feature vector for each tweet where each word is
    represented by integer index based on rank in vocabulary.
    """
    words = tweet.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def process_tweets(csv_file, test_file=True):
    """
    Generates training X, y pairs.
    """
    tweets = []
    labels = []
    print('Generating feature vectors')
    with open(csv_file, 'r', encoding="utf-8") as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
                # Convert sentiment labels to one-hot encoding
                sentiment_onehot = [0, 0, 0]
                if sentiment == "1":
                    sentiment_onehot[2] = 1
                elif sentiment == "0":
                    sentiment_onehot[1] = 1
                elif sentiment == "-1":
                    sentiment_onehot[0] = 1
                labels.append(sentiment_onehot)
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append(feature_vector)
            else:
                tweets.append(feature_vector)
            utils.write_status(i + 1, total)
    print('\n')
    return tweets, np.array(labels)


if __name__ == '__main__':
    np.random.seed(1337)
    vocab_size = 90000
    batch_size = 500
    max_length = 30
    filters = 600
    kernel_size = 3
    vocab = utils.top_n_words(FREQ_DIST_FILE, vocab_size, shift=1)
    
  
    tweets, labels = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    print(len(tweets), len(labels))
    glove_vectors = get_glove_vectors(vocab)
    # Create and embedding matrix
    embedding_matrix = np.random.randn(vocab_size + 1, dim) * 0.01
    # Seed it with GloVe vectors
    for word, i in vocab.items():
        glove_vector = glove_vectors.get(word)
        if glove_vector is not None:
            embedding_matrix[i] = glove_vector
    tweets = pad_sequences(tweets, maxlen=max_length, padding='post')
    shuffled_indices = np.random.permutation(tweets.shape[0])
    tweets = tweets[shuffled_indices]
    labels = labels[shuffled_indices]
    if train:
        model = Sequential()
        model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=max_length))
        model.add(Dropout(0.4))
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(300, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(150, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(75, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Flatten())
        model.add(Dense(600))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        # Change output layer to have 3 nodes and softmax activation
        model.add(Dense(3))
        model.add(Activation('softmax'))
        # Change loss function to categorical_crossentropy
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        filepath = "./models/4cnn-{epoch:02d}-{loss:0.3f}-{val_loss:0.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
        print(tweets, labels)
        model.fit(tweets, labels, batch_size=128, epochs=8, validation_split=0.1, shuffle=True, callbacks=[checkpoint, reduce_lr])
    else:
        model = load_model(MODEL_FILE)
        print(model.summary())
        test_tweets, _ = process_tweets(TEST_PROCESSED_FILE, test_file=True)
        test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
        predictions = model.predict(test_tweets, batch_size=128, verbose=1)
        results = np.argmax(predictions, axis=1) - 1 # Convert back to original labels (-1, 0, 1)
        id_results = zip(map(str, range(len(test_tweets))), results)
        utils.save_results_to_csv(id_results, 'cnn.csv')
        test_label = utils.file_number_to_list(TEST_LABEL_FILE)
        # Convert test labels to one-hot encoding
        test_label = pd.get_dummies(test_label).values
        # Remove the neutral column
        test_label = np.delete(test_label, 1, axis=1)
        report = classification_report(test_label, predictions, target_names=['negative', 'positive'], output_dict=True)
        print(classification_report(test_label, predictions, target_names=['negative', 'positive'], output_dict=False))
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(REPORT_FILE)
