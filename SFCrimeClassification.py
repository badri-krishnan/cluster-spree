import pandas as pd
import numpy as np

def clean_data(train, test):
    train_dataframe = pd.read_csv("input_data/train.csv")
    test_dataframe = pd.read_csv("input_data/test.csv")

    #Types of Crimes
    crime_types = {}
    unique_crimes = train_dataframe['Category'].unique()
    type_count = 1
    for types in unique_crimes:
        unique_crimes[types] = type_count
        type_count += 1
    train_dataframe['Category'] = train_dataframe['Category'].replace(unique_crimes)

    #Handle Day of Week conversion
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

    #Handling district types
    unique_district = train_dataframe['PdDistrict'].unique()
    district_count = 1
    district_dict = {}
    for district in unique_district:
        district_dict[district] = district_count
        district_count += 1

    train_dataframe['PdDistrict'] = train_dataframe['PdDistrict'].replace(district_dict)
    test_dataframe['PdDistrict'] = test_dataframe['PdDistrict'].replace(district_dict)

    return train_dataframe, test_dataframe


def main():
    training_set = 'input_data/train.csv'
    test_set = 'input_data/test.csv'

    train, test = clean_data(training_set, test_set)


if __name__ == "__main__":
    main()