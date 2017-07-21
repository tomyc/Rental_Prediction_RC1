"""
Build a predictive model using Python
Source: https://microsoft.github.io/sql-ml-tutorials/python/rentalprediction/
Tutorial changed to work with MS SQL Server 2017 RC1
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from revoscalepy.datasource.RxSqlData import RxSqlServerData
from revoscalepy.computecontext.RxInSqlServer import RxInSqlServer
from revoscalepy.etl.RxImport import rx_import #zmiana nazwy i definicji metordy
                                               #9.1.0 -> 9.2.0: rx_import_datasource -> rx_import

#Connection string to connect to SQL Server named instance
SQL_SERVER = os.getenv('RTEST_SQL_SERVER', '.')
CONN_STR = 'Driver=SQL Server;Server='+SQL_SERVER+';Database=TutorialDB;Trusted_Connection=True;'

#Define the columns we wish to import
COLUMN_INFO = {
    "Year" : {"type" : "integer"},
    "Month" : {"type" : "integer"},
    "Day" : {"type" : "integer"},
    "RentalCount" : {"type" : "integer"},
    "WeekDay" : {
        "type" : "factor",
        "levels" : ["1", "2", "3", "4", "5", "6", "7"]
        },
    "Holiday" : {
        "type" : "factor",
        "levels" : ["1", "0"]
        },
    "Snow" : {
        "type" : "factor",
        "levels" : ["1", "0"]
        }
    }

#Get the data from SQL Server Table
#data_source = RxSqlServerData(table="dbo.rental_data",
#connectionString=conn_str,
#colInfo=column_info)
#9.2.0 zmiana connectionString->connection_string; colInfo->column_info
DATA_SOURCE = RxSqlServerData(table="dbo.rental_data",
                              connection_string=CONN_STR,
                              column_info=COLUMN_INFO)

#zmiana w 9.2.0 connectionString -> connection_string;
#numTasks->num_tasks; autoCleanup-> auto_cleanup
COMPUTE_CONTEXT = RxInSqlServer(
    connection_string=CONN_STR,
    num_tasks=1,
    auto_cleanup=False
)

#RxInSqlServer(connectionString=conn_str, numTasks=1, autoCleanup=False)
#zmiana w 9.2.0
#connectionString -> connection_string; numTasks->num_tasks; autoCleanup-> auto_cleanup
RxInSqlServer(connection_string=CONN_STR, num_tasks=1, auto_cleanup=False)

# import data source and convert to pandas dataframe
#df = pd.DataFrame(rx_import_datasource(data_source))
DF = pd.DataFrame(rx_import(DATA_SOURCE)) #9.2.0 zmiana rx_import_datasource->rx_import
print("Data frame:", DF)
# Get all the columns from the dataframe.
COLUMNS = DF.columns.tolist()
# Filter the columns to remove ones we don't want to use in the training
COLUMNS = [c for c in COLUMNS if c not in ["Year"]]

# Store the variable we'll be predicting on.
TARGET = "RentalCount"
# Generate the training set.  Set random_state to be able to replicate results.
TRAIN = DF.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
TEST = DF.loc[~DF.index.isin(TRAIN.index)]
# Print the shapes of both sets.
print("Training set shape:", TRAIN.shape)
print("Testing set shape:", TEST.shape)
# Initialize the model class.
LIN_MODEL = LinearRegression()
# Fit the model to the training data.
LIN_MODEL.fit(TRAIN[COLUMNS], TRAIN[TARGET])

# Generate our predictions for the test set.
LIN_PREDICTIONS = LIN_MODEL.predict(TEST[COLUMNS])
print("Predictions:", LIN_PREDICTIONS)
# Compute error between our test predictions and the actual values.
LIN_MSE = mean_squared_error(LIN_PREDICTIONS, TEST[TARGET])
print("Computed error:", LIN_MSE)
