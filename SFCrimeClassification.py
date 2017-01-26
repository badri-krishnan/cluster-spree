import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict

'''
Frame the question we want to ask from the DataSet
1) From the District, Coordinates, and Day of The Week can find neighborhoods where particular crimes occur?(KNN)

'''


def clean_data():
    train_dataframe = pd.read_csv("input_data/train.csv")
    test_dataframe = pd.read_csv("input_data/test.csv")

    '''Types of Crimes'''
    crime_types = {}
    unique_crimes = train_dataframe['Category'].unique()
    crime_count = 1
    for types in unique_crimes:
        crime_types[types] = crime_count
        crime_count += 1

    ordered_crimes = OrderedDict(sorted(crime_types.items()))

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
    train_corr = train.corr()

    knn_model(train, test, ordered_crime)

if __name__ == "__main__":
    main()