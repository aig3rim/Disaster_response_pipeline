# Disaster_response_pipeline
A project on Data Engineering module for Data Scientist Nanodegree Program by [Udacity](https://www.udacity.com/) using data from [appen](https://appen.com/) (previously FigureEight)

## Goal
There are usually tons of overwhelming messages during natural disasters. The goal of this project is to build a data processing and interpretation pipeline to speed up the resource allocation process.

## Getting started
You need an installation of Python, plus the following libraries:
* numpy
* pandas
* sys
* sqlalchemy
* sklearn
* nltk
* warnings
* re
* pickle
* json
* plotly
* string
* flask

## How to run
1. Run the following commands in the project's root directory to set up your database and model.
* To run ETL pipeline that cleans data and stores in database
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
* To run ML pipeline that trains classifier and saves
`python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
