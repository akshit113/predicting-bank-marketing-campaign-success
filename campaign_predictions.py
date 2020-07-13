__author__ = 'Akshit Agarwal'
__email__ = 'akshit@email.arizona.edu'
__date__ = '2020-07-12'
__dataset__ = 'https://archive.ics.uci.edu/ml/datasets/bank+marketing'
__connect__ = 'https://www.linkedin.com/in/akshit-agarwal93/'

from pprint import pprint

import numpy as np
from pandas import get_dummies
from pandas import read_csv, DataFrame, concat, get_dummies
from sklearn.metrics import jaccard_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def import_dataset(fpath):
    """import .csv dataset into pandas dataset
    :param fpath: filepath of dataset
    :return: imported df
    """
    data = read_csv(fpath)
    print(data.head())
    print(data.shape)
    return data


def create_age_buckets(df):
    """divides age values (continuous) into bucket values (binning)
    :param df:
    :return: df with age buckets
    """
    print(df['age'].min())
    print(df['age'].max())
    old_values = df['age'].values.tolist()
    new_age_list = []
    new_age = 0
    for age in old_values:
        if age in range(18, 30):
            new_age = 1
        elif age in range(30, 40):
            new_age = 2
        elif age in range(40, 50):
            new_age = 3
        elif age in range(50, 65):
            new_age = 4
        elif age in range(65, 70):
            new_age = 5
        else:
            new_age = 6
        new_age_list.append(new_age)
    age_df = DataFrame(new_age_list, columns=['Age_Buckets'])
    df = concat([age_df, df], axis=1)
    return df


def check_unique_value(df, colnames):
    """Gets unique value counts for all selected columns in the dataframe including NaN values and PrettyPrints the
    dicitonary
    :param df:
    :param colnames:
    :return: a dictionary
    """
    mydict = {}
    for col in colnames:
        val_count = (df[col].value_counts(dropna=False)).to_dict()
        mydict[col] = val_count
    pprint(mydict)
    return


def normalize_columns(df, colnames):
    """Normalizes the specified columns in dataframe using min-max normalization
    :param df: df being normalized
    :param colnames: list of columns to be normalized
    :return: df with normalized columns
    """
    for col in colnames:
        s = df[col]
        df[col] = s.sub(s.min()).div((s.max() - s.min()))
    print(f'''Normalized Columns: {colnames}''')

    return df


def one_hot_encode(df, colnames):
    """This function performs one-hot encoding of the columns
    :param df: input df
    :param colnames: columns to be one-hot encoded
    :return: dataframe
    """
    for col in colnames:
        oh_df = get_dummies(df[col], prefix=col)
        df = concat([oh_df, df], axis=1)
        df = df.drop([col], axis=1)
    return df


def clean_dataset(df):
    """cleans dataset
    :param df: input dataset
    :return: cleaned dataset
    """
    df = create_age_buckets(df)
    print(df.columns)
    check_unique_value(df, ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome',
                            'y'])
    df = normalize_columns(df, ['balance', 'day', 'duration', 'campaign', 'pdays'])
    df = one_hot_encode(df, ['job', 'marital', 'education', 'default', 'housing',
                             'loan', 'contact', 'month', 'poutcome'])
    return df


def split_dataset(df, test_size, seed):
    """This function randomly splits (using seed) train data into training set and validation set. The test size
    paramter specifies the ratio of input that must be allocated to the test set
    :param df: one-hot encoded dataset
    :param test_size: ratio of test-train data
    :param seed: random split
    :return: training and validation data
    """
    ncols = np.size(df, 1)
    X = df.iloc[:, range(0, ncols - 1)]
    Y = df.iloc[:, ncols - 1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    y_train = get_dummies(y_train)  # One-hot encoding
    y_test = get_dummies(y_test)
    return x_train, x_test, y_train, y_test


def build_decision_tree(baseline=False):
    """Returns a decision tree classifier
    :param baseline: if true, returns a baseline decision tree ,otherwise returns the tree with specific hyperparamters
    :return: decision tree model
    """
    if baseline:
        model = DecisionTreeClassifier()
    else:
        model = DecisionTreeClassifier(criterion='entropy',
                                       splitter='best',
                                       max_depth=25)

    return model


def fit_decision_tree(model, x_train, y_train):
    """fits the model created in the get_model function on x_train, y_train and evaluates the model performance on
    x_test and y_test using the batch size and epochs paramters
    :param model: Sequential model
    :param x_train: training data
    :param y_train: training label
    :return: score, enumerated feature importance
    """
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    importance = model.feature_importances_
    return score, importance


def make_predictions(model, x_test, y_test):
    """Predicts on x_test, y_test using model
    :param model: decision tree model
    :param x_test: test features
    :param y_test: test labels
    :return: predicted and true labels
    """
    preds = model.predict(x_test)
    y_hat = np.argmax(preds, axis=-1)
    print(type(y_test))
    y_test.columns = [0, 1]
    y = y_test.idxmax(axis=1)
    print(y_hat.shape)
    print(y.shape)
    return y_hat, y


def calc_accuracy_using_metrics(y, y_hat, metric, average):
    """This function evaluates the model predictions, y_hat with ground truth y using a sklearn metric
    :param y: ground truth
    :param y_hat: model predictions
    :param metric: evaluation metric (f1_score, precision_score, jaccard_score
    :param average: micro, macro, binary, weighted
    :return: evaluation score
    """
    score = 0
    metrics_list = ['f1_score', 'jaccard_score', 'precision_score']
    average_list = ['micro', 'macro', 'binary']
    if metric not in metrics_list:
        print(f'''{metric} is not a valid metric type. Please try one of these: {metrics_list}''')
        return
    if average not in average_list:
        print(f'''{average} is not a valid average type. Please try one of these: {average_list}''')
        return
    if metric == 'f1_score':
        score = f1_score(y, y_hat, average=average)
    if metric == 'jaccard_score':
        score = jaccard_score(y, y_hat, average=average)
    if metric == 'precision_score':
        score = precision_score(y, y_hat, average=average)
    score = round(score, 4)
    print(f'''{metric}: {score}''')
    return score


def scores(y, y_hat):
    """Prints confusion matrix scores
    :param y:true labels
    :param y_hat: predicted labels
    :return:accuracy, misclassification, precision, recall, f_score
    """
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    accuracy = round((tp + tn) / (tp + tn + fp + fn), 3)
    misclassification = round((fp + fn) / (tp + tn + fp + fn), 3)
    precision = round(tp / (tp + fp), 3)
    recall = round(tp / (tp + fn), 3)
    f_score = round((2 * precision * recall) / (precision + recall), 4)
    return accuracy, misclassification, precision, recall, f_score


def main():
    fpath = r'C:\Users\akshitagarwal\Desktop\Keras\datasets\bank\bank\bank_data.csv'
    data = import_dataset(fpath)
    print(data.isna().sum())
    # no NA values found in the dataset
    df = clean_dataset(data)
    print(df.shape)
    colnames = list(df.columns)
    print(colnames)
    x_train, x_test, y_train, y_test = split_dataset(df, test_size=.2, seed=42)
    dec_tree = build_decision_tree()
    score, importance = fit_decision_tree(dec_tree, x_train, y_train)
    print(f'score is: {score}')
    y_hat, y = make_predictions(dec_tree, x_test, y_test)
    jcs = calc_accuracy_using_metrics(y, y_hat, metric='f1_score', average='binary')
    accuracy, misclassification, precision, recall, f_score = scores(y, y_hat)
    # print feature importance
    for i, v in enumerate(importance):
        print(f'Feature: {i}, {str(colnames[i])}, score: {round(v, 3)}')
    print(accuracy, misclassification, precision, recall, f_score)

    print('done')


if __name__ == '__main__':
    main()
