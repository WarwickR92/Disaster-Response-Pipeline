# import libraries
import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import pickle

# import nlp functions
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    loads in the file frpm the database and saves input variables for ml
    """
    # setup the database
    engine = create_engine('sqlite:///PipelineDatabase.db')

    # load in the file from the database
    df = pd.read_sql_table('Catagory_Data', engine) 

    # save the input variables for the ml
    X = df.message.values
    Y = df.iloc[:,4:].values

    # save the catagory column names to be used 
    catagories = df.iloc[:,4:].columns
    
    return X, Y, catagories

def tokenize(text):
    """
    This function returns a cleaned word list from text
    """
    # Removes punctuation from the text    
    text = re.sub(r'[^\w\s]','',text)

    # Changes text to lower case and splits to words 
    text = text.lower().split(' ')

    # Remove any blank strings    
    text = list(filter(None, text))

    # Remove stopwords from list
    text = [word for word in 
            text if word not in 
            stopwords.words('english')]

    return text


def build_model():
    # builds the ml pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [10, 50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    # create grid search object
    clf = GridSearchCV(pipeline, param_grid=parameters, n_jobs=2, verbose=2)

    return clf


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints f1 score, precision & recall for each catagory
    """
    # creates the predictions for the model on the test data
    y_pred = model.predict(X_test)

    # loops through all of the columns in the output data
    for column_no in range(0,len(Y_test[0])):
        
        # prints the catagory name        
        print(f'Catagory: {category_names[column_no]}')
        
        # prints the key evaluation metrics         
        test_values = [item[column_no] for item in Y_test]
        pred_values = [item[column_no] for item in y_pred]
        print(classification_report(test_values,pred_values))

    pass


def save_model(model, model_filepath):
    """
    saves the model for future use on the dataset
    """
    # saves the model to disk in the specified file path
    pickle.dump(model, open(model_filepath, 'wb'))
    
    pass


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