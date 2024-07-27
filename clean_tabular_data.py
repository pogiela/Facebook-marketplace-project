# Import the necessary libraries
import pandas as pd
import sys
import subprocess
import os

########## FUNCTIONS ##########
# function to clear the console
def clear_console():
    subprocess.run('cls' if os.name == 'nt' else 'clear', shell=True)

# Load the tabular data from the source file
def LoadTabularData(file_name):
    print(f'\n\n############## Loading data from source file: {file_name} ##############\n') 
    file_path = 'source_data/' + file_name
    
    try:
        data = pd.read_csv(
                    file_path, 
                    lineterminator='\n',
                    header=0
                )
    except Exception as e:
        print(f'Error occured when trying to filter the data: {e}')
        sys.exit()
    print('\n----> Success. Data loaded successfully\n')
    # print('----- Data information -----')
    # print(data.info())
    # print('\n----- Data sample -----')
    # print(data.head())
    return data
            
# function to remove the null values in any column or row
def RemoveNullValues(df):
    print('\n\n############## Cleaning null values ##############\n') 
    initial_columns = df.shape[1] # initial number of columns
    df = df.dropna(axis=1)
    final_columns = df.shape[1] # final number of columns
    print('----> Null values removed successfully. \n')
    print(f'  ----> {initial_columns - final_columns} columns removed\n')
    
    initial_rows = df.shape[0] # initial number of rows
    df = df.dropna(axis=0)
    final_rows = df.shape[0] # final number of rows
    print(f'  ----> {initial_rows - final_rows} rows removed\n')
    
    return df

# function to convert the prices into a numerical format
def ConvertPricesToNumericalFormat(df):
    print('\n\n############## Converting price into numerical format: ##############\n') 

    try:
        df['price'] = df['price'].replace(',', '', regex=True).replace('£', '', regex=True).astype(float)
    except Exception as e:
        print(f'Error occured: {e}')
        sys.exit()
    print('----> Product price column changed successfully\n')
    return df

# function to remove rows with £0.00 price
def RemoveRowsWithZeroPrice(df):
    print('\n\n############## Removing rows with £0.00 price ##############\n') 
    initial_rows = df.shape[0] # initial number of rows
    df = df[df['price'] != 0.0]
    final_rows = df.shape[0] # final number of rows
    print('----> Rows with £0.00 price removed successfully. \n')
    print(f'  ----> {initial_rows - final_rows} rows removed\n')
    return df



  
  

########## MAIN CODE ##########
# # Clear the console
# clear_console()

# # Load the tabular data from the source files
# images = LoadTabularData('Images.csv')
# products = LoadTabularData('Products.csv')

# # Clean the tabular data
# images = RemoveNullValues(images)
# products = RemoveNullValues(products)

# # Convert the prices into a numerical format
# products = ConvertPricesToNumericalFormat(products)

# # Remove rows with £0.00 price
# products = RemoveRowsWithZeroPrice(products)


def clean_tabular_data():
    # Load the tabular data from the source files
    images = LoadTabularData('Images.csv')
    products = LoadTabularData('Products.csv')

    # Clean the tabular data
    images = RemoveNullValues(images)
    products = RemoveNullValues(products)

    # Convert the prices into a numerical format
    products = ConvertPricesToNumericalFormat(products)

    # Remove rows with £0.00 price
    # products = RemoveRowsWithZeroPrice(products) <<-- Commented out because when the rows are removed, there are missing matches to the images dataset
    
    return products