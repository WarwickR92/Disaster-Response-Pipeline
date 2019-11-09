# Disaster Response Pipeline

This project is designed to create a web app where an emergency worker can input a new message and get a text classification results for 36 different categories. The web app will also display visualizations of the data so that decisions can be made based on the results.

The project loads, cleans & merges both the messages & catagories training data in python. This training data is used in a RandomForests machicne learning method to train a model to predict the classification on future messages. The new messages can be added into the web app and Flask is used to display the data.

To run the web app please use the following instructions:

### Instructions to run the web app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Explaination of files in the web app directory:
- README.md: Explains the context, directions for use and purpose of the web app.
- PipelineDatabase.db: An example database created by the web app containing training data.
- data/disaster_messages.csv: Example messages that can be used to train the machine learning model.
- data/disaster_categories.csv: Example categories that are linked to the above messages.
- data/process_data.py: The python file that loads, cleans & merges both the messages & catagories training data.
- models/train_classifier.py: The python file that trains the Random Forests ml model.
- app/run.py: The python file that runs the visualisations and creates the web app.
