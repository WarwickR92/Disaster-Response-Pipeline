import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Reads in and merges the messages and categories data
    """
    # read in the messages data from path
    messages =  pd.read_csv(messages_filepath)

    # read in the catagories data from path
    categories = pd.read_csv(categories_filepath)

    # merge both of the datasets based on the id
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """
    This function cleans the data ready for the ml process
    """
    # splits the catagories column into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Pulls the catagory types ready for re-naming the columns
    category_col_names = [x.split('-')[0] for x in categories.iloc[0,:]]

    # rename the all of the catagories columns
    categories.columns = category_col_names

    # convert the catagories data to integers of 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [x[-1] for x in categories[column]]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from dataframe
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1, sort=False)

    # findes all of the duplicates in the dataframe
    duplicates = df.duplicated()

    # removes duplicates from the dataframe
    df = df.loc[duplicates == False]

    return df


def save_data(df, database_filename):
    """
    creates the SQLite database and ouputs data
    """
    # creates the SQLite database
    engine = create_engine('sqlite:///PipelineDatabase.db')

    # saves dataframe to the database filename
    df.to_sql(database_filename, engine, index=False)

    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()