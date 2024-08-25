# Facebook Marketplace's Recommendation Ranking System and API
This project is divided into two categories:
1. Data processing, training Resnet50 neural network (Pytorch) and creating FAISS indexing for finding similar images.
2. Final trained model is used in an API in a Docker container which allows to find similar images.

## Built With

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

Version:
- Python 3.12.0

Other key technologies used:
- Pytorch (Resnet50)
- FAISS
- FastAPI
- Docker


## Downloading the project files:

1. Clone the repo
    ```
    git clone https://github.com/pogiela/Facebook-marketplace-project.git
    ```

## File structure of the project:
```
.
├── Facebook-marketplace-project            # Project files
    ├── app                                 # Folder containing API and Docker files
    |   ├── final_model                     # Folder containing the trained model parameters - ** NOT INCLUDED DUE TO SIZE **
    |   |   └── image_model.pt              # Final model parameters - ** NOT INCLUDED DUE TO SIZE **
    |   ├── api.py                          # API module to run a FastAPI for image recommendations
    |   ├── Dockerfile                      # Configuration file to build a Docker image
    |   ├── faiss_index.index               # FAISS Index file - ** NOT INCLUDED DUE TO SIZE **
    |   ├── functions.py                    # Module containing functions that are used in other file
    |   ├── image_decoder.pkl               # Dictionary file containing decoder and encoder - ** NOT INCLUDED DUE TO SIZE **
    |   ├── image_embeddings.json           # Image embeddings - ** NOT INCLUDED DUE TO SIZE **
    |   ├── image_processor.py              # Module containing a function to process image as an input and apply transfomations needed for the model
    |   ├── requirements.txt                # List of dependiencies to be installed for the Docker image
    ├── source_code                         # Folder containing source code for data preparation and model training
    |   ├── source_data                     # Folder containing source data - ** NOT INCLUDED DUE TO SIZE **
    |   |   ├── images                      # Folder containing source images corresponding to the products from the dataset
    |   |   ├── Images.csv                  # CSV file containing a list of all images IDs and corresponding product's ID (id, product_id)
    |   |   └── Products.csv                # CSV file containing a list of products including its price, location, and description.
    |   ├── test_images                     # Folder containing final custom test images
    |   |   └── laptop.webp                 # Test image
    |   ├── 1. Prepare Data.ipynb           # This notebook prepares images for the model training
    |   ├── 2. Train Model.ipynb            # This notebook trains the model
    |   ├── 3. Embeddings Extraction.ipynb  # This notebook is extracting embeddings for all images
    |   ├── 3. Embeddings Extraction.ipynb  # This notebook creates FAISS index based on the embeddings
    |   ├── clean_images.py                 # Module containing functions helping to clean and prepare images for training
    |   ├── clean_tabular_data.py           # Module containing functions helping to clean the tabular dataset and create training dataset
    |   ├── create_pytorch_dataset.py       # Module containing the class to create a custom dataset for the images
    |   ├── functions.py                    # Module containing functions that are used in other files
    |   ├── image_processor.py              # Module containing a function to process image as an input and apply transfomations needed for the model        
    └── README.md                       # This file
```

## Usage instructions:

1. Create new conda environment and activate it
    ```
    conda create -n facebook_marketplace pip pandas matplotlib numpy
    conda activate facebook_marketplace
    ```

2. Install all dependencies listed below. If using Conda environment use the *conda* commands (where available) or use *pip* commands if not using a dedicated conda environment.
    1. torch: conda install torch | pip install torch
    2. torchvision: conda install torchvision | pip install torchvision
    3. tensorboard: conda install tensorboard | pip install tensorboard
    4. faiss-cpu: conda install faiss-cpu | pip install faiss-cpu
    5. uvicorn: conda install uvicorn pip | install uvicorn
    6. fastapi: conda install fastapi | pip install fastapi
    7. python-multipart: conda install python-multipart | pip install python-multipart


## SECTION 1 - Data preparation, model training and creating index

### Usage instructions:

1. Enter *Facebook-marketplace-project/source_code* folder:
    ```
    cd ./Facebook-marketplace-project/source_code
    ```

2. To process the data, train model and create FAISS search index, run the following notebooks in the order:
    - *1. Prepare Data.ipynb* - This notebook prepares images for the model training
    - *2. Train Model.ipynb* - This notebook trains the model
    - *3. Embeddings Extraction.ipynb* - This notebook is extracting embeddings for all images
    - *3. Embeddings Extraction.ipynb* - This notebook creates FAISS index based on the embeddings


## SECTION 2 - API

### API Endpoints

1. GET /healthcheck
   - Healthcheck endpoint to verify that the API is up and running.
   - Returns a simple JSON message indicating the status of the API.

2. POST /predict/feature_embedding
   - Accepts an image file and returns its feature embeddings.
   - Expects: An image file in the request body.
   - Returns: A JSON response containing the feature embeddings of the image.

3. POST /predict/similar_images
   - Accepts an image file and a text input, then finds the top 5 most similar images to the uploaded image.
   - Expects: An image file in the request body and an optional text input.
   - Returns: A JSON response containing the indices and distances of the top 5 similar images.

### Usage instructions:

1. Enter *Facebook-marketplace-project/api* folder:
    ```
    cd ./Facebook-marketplace-project/api
    ```

2. Run the API locally:
    ```
    uvicorn api:app --host 0.0.0.0 --port 8080 --reload
    ```

3. Test locally if the API is running:
    ```
    curl http:localhost:8080/healthcheck
    ```

4. Create docker image:
    ```
    docker buildx build --platform linux/amd64 -t facebook_marketplace_recommendations .
    ```

5. Run the docker image:
    ```
    docker run -p 8080:8080 -it facebook_marketplace_recommendations:latest
    ```

## LIVE VERSION EXAMPLE
1. A live version of the API is available at the address: http://ec2-13-40-44-15.eu-west-2.compute.amazonaws.com:8080/docs

2. Example similar images search:
![alt text](doc_images/sofa.png)


## License information:
Distributed under the MIT License. 

## Contact
Piotr Ogiela - piotrogiela@gmail.com

Project Link: https://github.com/pogiela/Facebook-marketplace-project.git






