'''
This file contains the functions that are used in other files

Functions:
    clear_console: function to clear the console
        parameters: None
        
    progress: function to show the progress bar
        parameters: count: integer, total: integer
'''

import sys
import subprocess
import os
import pickle

# function to clear the console
def clear_console():
    '''
    Clears the console screen
    '''
    subprocess.run('cls' if os.name == 'nt' else 'clear', shell=True)

# function to show the progress bar
def progress(count, total):
    '''
    Displays a progress bar with % completed and the number 
    of processed items and total number of items to process
    
    Parameters:
    -----------
    count: number
        Number of current item to process
    total: number
        Number of total items to process
    '''
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f'[{bar}] {percents}%  [{count} / {total}]\r')
    sys.stdout.flush()
    

def LoadEncoderDecoder(filename):
    '''
    Function to load the encoder and decoder from a file
    '''
    # Load decoder dictionary from file
    print('\n\n########## Load Decoder Dictionary ##########')
    with open(filename, "rb") as f:
        decoder_dict = pickle.load(f)
    print('----> Decoder dictionary loaded successfully')
    
    # Extract the encoder and decoder from the loaded dictionary
    encoder = decoder_dict['encoder']
    decoder = decoder_dict['decoder']
    print('----> Encoder and decoder extracted from the dictionary')
    print(f"----> Encoder:\n {encoder}")
    print(f"----> Decoder:\n {decoder}")
    
    return encoder, decoder
    