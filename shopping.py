import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")



def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer*
        - Administrative_Duration, a floating point number*
        - Informational, an integer*
        - Informational_Duration, a floating point number*
        - ProductRelated, an integer*
        - ProductRelated_Duration, a floating point number*
        - BounceRates, a floating point number*
        - ExitRates, a floating point number*
        - PageValues, a floating point number*
        - SpecialDay, a floating point number*
        - Month, an index from 0 (January) to 11 (December)******
        - OperatingSystems, an integer*
        - Browser, an integer*
        - Region, an integer*
        - TrafficType, an integer*
        - VisitorType, an integer 0 (not returning) or 1 (returning)*
        - Weekend, an integer 0 (if false) or 1 (if true)*

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    
    #  load data file
    df = pd.read_csv('shopping.csv') 

    # prepare data for the nearest-neighbor classifier, all of our data needs to be numeric
    # convert all values in the  columns to numeric values
    df['Administrative'] = pd.to_numeric(df['Administrative'], downcast="integer")
    df['Administrative_Duration'] = pd.to_numeric(df['Administrative_Duration'], downcast='float')
    df['Informational'] = pd.to_numeric(df['Informational'], downcast="integer")
    df['Informational_Duration'] = pd.to_numeric(df['Administrative_Duration'], downcast='float')
    df['ProductRelated'] = pd.to_numeric(df['ProductRelated'], downcast="integer")
    df['ProductRelated_Duration'] = pd.to_numeric(df['ProductRelated_Duration'], downcast='float')
    df['BounceRates'] = pd.to_numeric(df['BounceRates'], downcast='float')
    df['ExitRates'] = pd.to_numeric(df['ExitRates'], downcast='float')
    df['PageValues'] = pd.to_numeric(df['PageValues'], downcast='float')
    df['SpecialDay'] = pd.to_numeric(df['SpecialDay'], downcast='float')

    df['OperatingSystems'] = pd.to_numeric(df['OperatingSystems'], downcast="integer")
    df['Browser'] = pd.to_numeric(df['Browser'], downcast="integer")
    df['Region'] = pd.to_numeric(df['Region'], downcast="integer")
    df['TrafficType'] = pd.to_numeric(df['TrafficType'], downcast="integer")

    for x in df.index:
        if df.loc[x, "VisitorType"] ==  'Returning_Visitor':
            df.loc[x, "VisitorType"] = 1
        else:
            df.loc[x, "VisitorType"] = 0

    for x in df.index:
        if df.loc[x, "Weekend"] ==  False:
            df.loc[x, "Weekend"] = 0
        else:
            df.loc[x, "Weekend"] = 1

    for x in df.index:
        if df.loc[x, "Revenue"] ==  True:
            df.loc[x, "Revenue"] = 1
        else:
            df.loc[x, "Revenue"] = 0


    # normalize the Month column  
    for x in df.index:
        if df.loc[x, "Month"] ==  'June':
            df.loc[x, "Month"] = 'Jun'
            

    # create a translation dictionary to convert months to a numeric value
    monthDict = {
    "Jan": 0,
    "Feb": 1,
    "Mar": 2,
    "Apr": 3,
    "May": 4,
    "Jun": 5,
    "Jul": 6,
    "Aug": 7,
    "Sep": 8,
    "Oct": 9,
    "Nov": 10,
    "Dec": 11,
} 

    #create a list of Lable data
    for x in df.index:
        df.loc[x, "Month"] = monthDict[df.loc[x, "Month"]]

    #create a list of label data for the nearest-neighbor model
    labels_list=list(df['Revenue'])

    # create a list of evidence values
    del df['Revenue']
    evidence_list = df.values.tolist()

    training_data = (evidence_list,labels_list)


    
    return training_data


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    
    # determine how many times a purchase was actual compleated
    actual_correct_count = 0
    x = 0
    while x in range(len(labels)):
        if labels[x] == 1:
            actual_correct_count +=1
        x+=1 
    
    # determine how many times the model predicted correctly that a purchase was completed
    predicted_correct_count = 0
    x = 0
    while x in range(len(labels)):
        if predictions[x] == labels[x] and labels[x] == 1:
            predicted_correct_count +=1
        x+=1 
        
    # calculate the proportion of correct predictions vs the actual value   
    sensitivity = predicted_correct_count / actual_correct_count


    # determine how many times a purchase was not completed
    actual_incorrect_count = 0
    x = 0
    while x in range(len(labels)):
        if labels[x] == 0:
            actual_incorrect_count +=1
        x+=1 
    
    # determine how many times the model predicted correctly that a purchase was not completed
    predicted_incorrect_count = 0
    x = 0
    while x in range(len(labels)):
        if predictions[x] == labels[x] and labels[x] == 0:
            predicted_incorrect_count +=1
        x+=1 
    
    # calculate the proportion of correct predictions vs the actual value 
    specificity = predicted_incorrect_count / actual_incorrect_count


    S_S_tpl = (sensitivity, specificity)


    return S_S_tpl


if __name__ == "__main__":
    main()
