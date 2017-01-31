import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict
from matplotlib import pyplot as plt

'''
Frame the question we want to ask from the DataSet
1) From the District, Coordinates, and Day of The Week can find neighborhoods where particular crimes occur?(KNN)
2) Can we predict more serious crimes based on the hour, District, X, Y, - Model 2
'''
def categorize_crimes(crime_type):
    red_crimes = ['ARSON', 'ASSAULT', 'BURGLERY', 'ROBBERY', 'MISSING PERSON', 'RUNAWAY', 'SEX OFFENSES NON FORCIBLE',
              'SUICIDE', 'WEAPON LAWS', 'KIDNAPPING', 'VEHICLE THEFT', 'WARRANTS', 'PROSTITUTION', 'RUNAWAY','SUSPICIOUS OCC']

    yellow_crimes = ['DRUG/NARCOTIC', 'LARCENY/THEFT', 'EMBEZZLEMENT', 'FRAUD', 'DISORDERLY CONDUCT', 'VANDALISM', 'EXTORTION',
                         'PORNOGRAPHY/OBSCENE MAT', 'FORGERY/COUNTERFEITING', 'STOLEN PROPERTY', 'DRIVING UNDER THE INFLUENCE']

    white_crimes = ['BAD CHECKS', 'BRIBERY', 'DISORDERLY CONDUCT', 'DRUNKENNESS', 'FAMILY OFFENSES', 'GAMBLING',
                    'LIQUOR LAWS', 'LOITERING', 'NON-CRIMINAL', 'OTHER OFFENSES', 'TRESPASS', 'TREA', 'SECONDARY CODES']
    if crime_type in red_crimes:
        return 1
    elif crime_type in yellow_crimes:
        return 2
    elif crime_type in white_crimes:
        return 3
    else:
        return 3
def convert_date_to_year(dateobj):
    year = dateobj.year
    return year

def convert_date_to_month(dateobj):
    month = dateobj.month
    return month

def convert_date_to_day(dateobj):
    day = dateobj.day
    return day

def convert_date_to_hours(dateobj):
    hour = dateobj.hour
    return hour

def clean_data():
    train_dataframe = pd.read_csv("input_data/train.csv")
    test_dataframe = pd.read_csv("input_data/test.csv")

    print train_dataframe.head(2)
    '''Types of Crimes'''
    crime_types = {}
    unique_crimes = train_dataframe['Category'].unique()
    crime_count = 1
    for types in unique_crimes:
        crime_types[types] = crime_count
        crime_count += 1

    ordered_crimes = OrderedDict(sorted(crime_types.items()))

    '''Create the CrimeRating, Kind of a dubious stat for now'''
    train_dataframe['CrimeRating'] = train_dataframe['Category'].apply(categorize_crimes)

    train_dataframe['Category'] = train_dataframe['Category'].replace(crime_types)

    '''Handle Day of Week conversion'''
    day_dict = {
        'Monday':1,
        'Tuesday':2,
        'Wednesday':3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7,
    }
    train_dataframe['DayOfWeek'] = train_dataframe['DayOfWeek'].replace(day_dict)
    test_dataframe['DayOfWeek'] = test_dataframe['DayOfWeek'].replace(day_dict)

    train_dataframe['Dates'] = pd.to_datetime(train_dataframe['Dates'])
    test_dataframe['Dates'] = pd.to_datetime(test_dataframe['Dates'])

    '''Handling the splitting up of dates and to Year, hours, and Dates'''
    train_dataframe['Year'] = train_dataframe['Dates'].apply(convert_date_to_year)
    train_dataframe['Month'] = train_dataframe['Dates'].apply(convert_date_to_month)
    train_dataframe['Day'] = train_dataframe['Dates'].apply(convert_date_to_day)
    train_dataframe['Hour'] = train_dataframe['Dates'].apply(convert_date_to_hours)

    test_dataframe['Year'] = test_dataframe['Dates'].apply(convert_date_to_year)
    test_dataframe['Month'] = test_dataframe['Dates'].apply(convert_date_to_month)
    test_dataframe['Day'] = test_dataframe['Dates'].apply(convert_date_to_day)
    test_dataframe['Hour'] = test_dataframe['Dates'].apply(convert_date_to_hours)



    '''Handling district types'''
    unique_district = train_dataframe['PdDistrict'].unique()
    district_count = 1
    district_dict = {}
    for district in unique_district:
        district_dict[district] = district_count
        district_count += 1

    train_dataframe['PdDistrict'] = train_dataframe['PdDistrict'].replace(district_dict)
    test_dataframe['PdDistrict'] = test_dataframe['PdDistrict'].replace(district_dict)

    columns = train_dataframe.columns.drop("Resolution")
    train_dataframe = train_dataframe[columns]
    return train_dataframe, test_dataframe, ordered_crimes

def model_2(train, test, crime_dict):
    features = ['PdDistrict', 'X', 'Y', 'Hour']

    X_train = train[features]
    X_test = test[features]
    Y_train = test[features]



def knn_model(train, test, crime_dict):
    features = ['DayOfWeek', 'PdDistrict', 'X', 'Y']
    X_train = train[features]
    Y_train = train['Category']
    X_test = test[features]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)
    Y_test_predictions = knn.predict(X_test)
    print Y_test_predictions
    predict_dataframe = pd.DataFrame({
        "Id": test["Id"]
    })

    for k,v in crime_dict.items():
        print k,v
        predict_dataframe[k] = 0
    count = 0
    for prediction in Y_test_predictions:
        for key,value in crime_dict.items():
            if(value == prediction):
                predict_dataframe[key][count] = 1
        count+=1

    predict_dataframe.to_csv(path_or_buf="output_data/KNN_Analysis.csv", index=True)

def main():
    training_set = 'input_data/train.csv'
    test_set = 'input_data/test.csv'

    train, test, ordered_crime = clean_data()
    print ordered_crime

    print "\nPrinting training data correlation matrix"
    cor_mat = train.corr()
    print cor_mat

    print "\nPrinting training data covariance matrix"
    cov_mat = train.cov()
    print cov_mat



   #knn_model(train, test, ordered_crime)
    model_2(train, test, ordered_crime)

if __name__ == "__main__":
    main()