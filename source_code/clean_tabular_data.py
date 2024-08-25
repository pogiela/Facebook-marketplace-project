'''
This module contains the functions that are used to load and clean tabular data

Functions:
    LoadTabularData: function to load the tabular data from the source file
        parameters: file_name: string
        
    RemoveNullValues: function to remove the null values in any column or row
        parameters: df: pandas dataframe
        
    ConvertPricesToNumericalFormat: function to convert the prices into a numerical format
        parameters: df: pandas dataframe
        
    RemoveRowsWithZeroPrice: function to remove rows with £0.00 price
        parameters: df: pandas dataframe
    
    LoadAndCleanData: function to load and clean the data from the source files
        parameters: None
        
    AddLabelColumn: function to add a new column 'label' to the products data which extracts the root category from the 'category' column
        parameters: products: pandas dataframe
        
    CreateEncoderDecoder: function to create an encoder and decoder for the 'label' column
        parameters: products: pandas dataframe
        
    SaveEncoderDecoder: function to save the encoder and decoder to a file
        parameters: filename: string, category_encoder: dictionary, category_decoder: dictionary
        
    MergeImagesAndProductsData: function to merge the images and products dataframes
        parameters: images: pandas dataframe, products: pandas dataframe
        
    SaveTrainingData: function to save the training data to a file
        parameters: data: pandas dataframe, target_folder: string
'''
import pandas as pd
import sys
import os
import pickle

########## FUNCTIONS ##########

# Load the tabular data from the source file
def LoadTabularData(file_name):
    '''
    Load the tabular data from the source file
    '''
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
    print('\n----- Data shape -----')
    print(data.shape)
    print('\n----- Data information -----')
    print(data.info())
    print('\n----- Data sample -----')
    print(data.head())
    return data
            
# function to remove the null values in any column or row
def RemoveNullValues(df):
    '''
    Remove the null values in any column or row
    '''
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
    '''
    Convert the prices into a numerical format
    '''
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
    '''
    Remove rows with £0.00 price
    '''
    print('\n\n############## Removing rows with £0.00 price ##############\n') 
    initial_rows = df.shape[0] # initial number of rows
    df = df[df['price'] != 0.0]
    final_rows = df.shape[0] # final number of rows
    print('----> Rows with £0.00 price removed successfully. \n')
    print(f'  ----> {initial_rows - final_rows} rows removed\n')
    return df


def LoadAndCleanData():
    '''
    Load and clean the data from the source files
    '''
    #### IMAGES DATA ####
    # Load the images data from the source files
    images = LoadTabularData('Images.csv')
    # Remove null values
    images = RemoveNullValues(images)

    #### PRODUCTS DATA ####
    # Load the products data from the source files
    products = LoadTabularData('Products.csv')
    # Remove null values
    products = RemoveNullValues(products)
    # Convert the prices into a numerical format
    products = ConvertPricesToNumericalFormat(products)
    
    return images, products

def AddLabelColumn(products):
    '''
    Add a new column 'label' to the products data which extracts the root category from the 'category' column
    '''
    print('############## Adding a new column "label" to the products data ##############\n')
    products['label'] = products['category'].str.split(' / ').str[0]
    print('----> New column "label" added successfully\n')
    return products

def CreateEncoderDecoder(products):
    '''
    Create an encoder and decoder for the 'label' column
    '''
    # Create an encoder for the 'label' column (category to numerical value)
    print('############## Creating encoder for the label data ##############\n')
    category_encoder = {label: idx for idx, label in enumerate(products['label'].unique())}
    print('----> Encoder created successfully\n')
    print('----> Encoder:\n', category_encoder, '\n')
    # Create a decoder (number to category)
    print('############## Creating decoder for the label data ##############\n')
    category_decoder = {idx: category for category, idx in category_encoder.items()}
    print('----> Decoder created successfully\n')
    print('----> Decoder:\n', category_decoder, '\n')
    
    return category_encoder, category_decoder

def SaveEncoderDecoder(filename, category_encoder, category_decoder):
    '''
    Save the encoder and decoder to a file
    '''
    print('############## Saving the encoder and decoder to a file ##############\n')
    # Combine the encoder and decoder into a single dictionary for convenience
    data = {'encoder': category_encoder, 'decoder': category_decoder}
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f'----> Encoder and decoder saved successfully to {filename}\n')

    return

def MergeImagesAndProductsData(images, products, encoder):
    '''
    Merge the images and products dataframes
    '''
    print('############## Merging the images and products dataframes ##############\n')
    # Merge images with products on product_id and id
    data = pd.merge(images, products[['id', 'label']], left_on='product_id', right_on='id', how='left')

    # Drop the extra id column
    data.drop(columns=['id_y'], inplace=True)

    # Rename the columns for clarity
    data.rename(columns={'Unnamed: 0': 'index', 'id_x': 'image_id', 'id': 'product_id'}, inplace=True)

    print('----> Data merged successfully\n')
    
    print('############## Encoding the labels ##############\n')
    data['label'] = data['label'].map(encoder)
    print('----> Labels encoded successfully\n')
    
    # Display the merged DataFrame
    print(data.head())
    
    return data

def SaveTrainingData(data, target_folder, filename):
    '''
    Save the training data to a file
    '''
    print('\n############## Saving the training data to a file ##############\n')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f'----> Target folder created successfully: {target_folder}\n')
    data.to_csv(target_folder + '/' + filename, index=False)

    print(f'----> Success. The training data has been saved to .csv file: {target_folder}/{filename}  \n')

    return

if __name__ == "__main__":
    print('This is a module containing functions to load and clean tabular data')
    print('Please run the main programme to execute the data processing')
    sys.exit()