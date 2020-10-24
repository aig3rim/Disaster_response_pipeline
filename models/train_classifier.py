# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')

def load_data(database_filepath):
    '''
    This function loads data from database_filepath.

    Parameters:
    database_filepath (str):  a path to the database

    Returns:
    X (Series): features
    y (DataFrame): target
    category_names (list): category names
    '''

    # create an engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # read the data into a pandas dataframe
    df = pd.read_sql_table('disaster_response', engine)

    # define features
    X = df['message']

    # define target
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)

    # define labels
    category_names = list(np.array(y.columns))

    return X, y, category_names


def tokenize(text):
    '''
    Tokenization function to process data.

    Parameters:
    text (pandas Series): text to be processed

    Returns:
    clean_tokens (list): list of words converted to a lower case with removed stop words
    '''

    # Converting everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize words
    tokens = word_tokenize(text)

    # normalization word tokens and remove stop words
    normlizer = PorterStemmer()
    stop_words = stopwords.words("english")

    clean_tokens = [normlizer.stem(word) for word in tokens if word not in stop_words]

    return clean_tokens


def build_model():
    '''
    Function builds a pipeline with GridSearch.

    Returns:
    cv: model (product of GridSearch).

    '''

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Apply grid search
    parameters = {'vect__min_df': [5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[60],
              'clf__estimator__max_depth': [10, 15],
              'clf__estimator__n_jobs': [-1]}

    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs=-1, verbose = 48)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Evaluate performance of ML pipeline on test data.

    Parameters:
    model: result of build_model function
    X_test: test features
    y_test: target for the test dataset
    category_names: names of categories

    Returns:
    precision, recall, fscore: evaluation metrics

    '''
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col],
                                                                    y_pred[:, i],
                                                                    average='weighted')

        print('\nReport for the column ({}):\n'.format(colored(col, 'red', attrs=['bold', 'underline'])))

        if precision >= 0.75:
            print('Precision: {}'.format(colored(round(precision, 2), 'green')))
        else:
            print('Precision: {}'.format(colored(round(precision, 2), 'yellow')))

        if recall >= 0.75:
            print('Recall: {}'.format(colored(round(recall, 2), 'green')))
        else:
            print('Recall: {}'.format(colored(round(recall, 2), 'yellow')))

        if fscore >= 0.75:
            print('F-score: {}'.format(colored(round(fscore, 2), 'green')))
        else:
            print('F-score: {}'.format(colored(round(fscore, 2), 'yellow')))

def save_model(model, model_filepath):
    '''
    Save model to a pickle file
    ''''

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
