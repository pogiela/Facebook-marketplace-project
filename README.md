# Facebook Marketplace's Recommendation Ranking System
TODO: add description


## Built With

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

Version:
- Python 3.12.0

## Installation instructions:

1. Clone the repo
    ```
    git clone https://github.com/pogiela/Facebook-marketplace-project.git
    ```

## Usage instructions:

1. Enter *Facebook-marketplace-project* folder:
    ```
    cd ./Facebook-marketplace-project/
    ```

2. Install all dependencies listed below. If using Conda environment use the *conda* commands (where available) or use *pip* commands if not using a dedicated conda environment.
    <!-- 1. yaml: conda install pyyaml | pip3 install pyyaml
    2. sqlalchemy: conda install sqlalchemy | install sqlalchemy
    3. pandas: conda install pandas | pip3 install pandas
    4. tabula: conda install -c conda-forge tabula-py | pip3 install tabula-py
    5. fitz: pip3 install PyMuPDF
    6. requests: conda install requests | pip3 install requests
    7. pathlib: conda install pathlib | pip3 install pathlib
    8. psycopg2: conda install psycopg2 | pip3 install psycopg2
    9. dotenv: conda install python-dotenv | pip3 install python-dotenv
    10. boto3: conda install boto3 | pip3 install boto3
    11. beautifultable: conda install beautifultable --channel conda-forge | pip3 install beautifultable -->

3. Add environmental variables and database credential files (see sections Environmental Variables and Database Connection Details).

4. To run the programme enter command:
    <!-- ```
    python3 ./start_data_processing.py 
    ``` -->

## File structure of the project:
<!-- ```
.
├── multinational-retail-data-centralisation156     # Project files
   ├── .env                                        # FILE NOT INCLUDED IN REPO: Environmental variables - see section Environmental Variables.
   ├── .db_creds.yaml                              # FILE NOT INCLUDED IN REPO: Database connection details - see section Database Connection Details.
   ├── data_cleaning.py                            # DataCleaning class and methods helping to clean the data before uploading to the database.
   ├── data_extraction.py                          # DataExtractor class and methods helping to extract data from various data sources.
   ├── data_processing.py                          # DataProcessing class and methods helping to extract data from various data sources. Parent class to DataExtractor.
   ├── database_query.py                           # DatabaseQuery class and methods used to query the database.
   ├── database_schema.py                          # DatabaseSchema class and methods helping to create star schema.
   ├── database_utils.py                           # DatabaseConnector class and methods helping to connect to and upload data to a database.
   ├── queries_data.sql                            # SQL Queries used to query the database.
   ├── queries_table_alterations.sql               # SQL Queries used to alter database tables to create star schema.
   ├── README.md                                   # This file
   └── start_data_processing.py                    # Main programme. Run this file to start the process.
``` -->

## Environmental Variables
<!-- The programme to run properly requires .env file to be created with the following fields. This file is not included in the repository and need to be created by the user.

- *CARD_DETAILS_DATA* - url to a PDF file containing Card Details Data.
- *DATE_EVENTS_DATA* - url to a JSON file containing Date Events Data.
- *PRODUCTS_DATA* url to a CSV file containing Products Data.
- *x_api_key* - API connection key
- *retrive_store_api* - API URL to get Stores Details Data
- *number_of_stores_api* - API URL to get Number of Stores -->

## Database Connection Details
<!-- The programme to run properly requires .db_creds.yaml file to be created with the following fields. This file is not included in the repository and need to be created by the user. The file needs to have two sections - one for Source database and another one for Output database. The file need to be written in the below format:

- *SOURCE:*
    - *RDS_HOST:* - RDS database host url
    - *RDS_PASSWORD:* Password to the database
    - *RDS_USER:* User to the database
    - *RDS_DATABASE:* Database name
    - *RDS_PORT:* Port number (default is *5432*)
    - *DATABASE_TYPE:* Type of the database (defaul is *postgresql*)
    - *DBAPI:* Database API type (default is *psycopg2*)

- *OUTPUT:*
    - *RDS_HOST:* - RDS database host url
    - *RDS_PASSWORD:* Password to the database
    - *RDS_USER:* User to the database
    - *RDS_DATABASE:* Database name
    - *RDS_PORT:* Port number (default is *5432*)
    - *DATABASE_TYPE:* Type of the database (defaul is *postgresql*)
    - *DBAPI:* Database API type (default is *psycopg2*)
 -->

## License information:
Distributed under the MIT License. 

## Contact
Piotr Ogiela - piotrogiela@gmail.com

Project Link: https://github.com/pogiela/Facebook-marketplace-project.git






