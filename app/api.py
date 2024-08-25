"""
api.py
This script implements a FastAPI web service that provides two primary functionalities:
1. Extracting feature embeddings from images using a pre-trained ResNet-50 model.
2. Finding the top 5 most similar images to a given image using a FAISS index.

Author: Piotr Ogiela
Date: 25/08/2024

Endpoints:
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
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import File, UploadFile, Form
import torch
import torch.nn as nn
from torchvision import models
import faiss
import json
import numpy as np
from image_processor import process_image
from functions import LoadEncoderDecoder

########## VARIABLES ##########
encoder_dictionary_filename = 'image_decoder.pkl'

# Define the Feature Extraction model
class FeatureExtractor(nn.Module):
    """
    FeatureExtractor model class.
    Uses a pre-trained ResNet-50 model architecture with custom modifications to extract feature embeddings from images.
    """
    def __init__(self, decoder: dict = None):
        super(FeatureExtractor, self).__init__()

        print("########## Initialising Feature Extractor model ##########")
        
        # Recreate the model architecture
        # Load ResNet-50 pre-trained model architecture without pre-trained weights
        self.main = models.resnet50(weights=None)
        print('----> Model architecture loaded successfully')
        
        # Get the number of features from the last fully connected layer
        self.num_features = self.main.fc.in_features
        print(f"----> Number of features: {self.num_features}")
        
        # Replace the final fully connected layer with a custom layer
        # The custom layer includes a dropout for regularization and reduces the dimensions to 1000
        self.main.fc = nn.Sequential(
            nn.Linear(self.num_features, 1024), # Reduce dimensions
            nn.ReLU(),                          # ReLU activation
            nn.Dropout(p=0.5),                  # Dropout layer to prevent overfitting
            nn.Linear(1024, 1000)               # Final linear layer to match the number of classes
        )
        print('----> Final linear layer replaced with required number of classess')
        
        # Modify the final layer
        self.main.fc = nn.Linear(self.num_features, 1000) # Change the number of output features to 1000
        print("----> Last layer updated")
        
        # Load the pre-trained model weights
        self.model_weights_path = 'final_model/image_model.pt'
        self.main.load_state_dict(torch.load(self.model_weights_path, map_location=torch.device('cpu')))
        print("----> Model weights loaded successfully.")
        
        # Initialize the decoder for class mappings
        self.decoder = decoder
        print("----> Decoder loaded.\n")

    def forward(self, image):
        """
        Forward pass for feature extraction.
        """
        x = self.main(image)
        return x

    def predict(self, image):
        """
        Extracts feature embeddings from the input image without computing gradients.
        """
        with torch.no_grad():
            x = self.forward(image)
            return x

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    """
    Pydantic model for receiving text input in API requests.
    """
    text: str


# Initialize the Feature Extraction model
try:
     # Load the encoder and decoder dictionaries
    encoder, decoder = LoadEncoderDecoder(encoder_dictionary_filename)
    num_classes = len(decoder)
    print(f"Number of classes: {num_classes}")
    
    # Initialize the Feature Extraction model
    model = FeatureExtractor(decoder)
    model.eval()
except Exception as e:
    raise OSError(f"No Feature Extraction model found. Check that you have the decoder and the model in the correct location: {str(e)}")
    

# Load the FAISS index for similarity search
try:
    faiss_index = faiss.read_index('faiss_index.index')
    with open('image_embeddings.json', 'r') as f:
        image_embeddings = json.load(f)     # Load the image embeddings
    image_ids = list(image_embeddings.keys())
except Exception as e:
    raise OSError(f"No Image model found. Check that you have the encoder and the model in the correct location: {str(e)}")


# Initialize FastAPI app
app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
    """
    Healthcheck endpoint to verify that the API is up and running.
    Returns a simple JSON message.
    """
    msg = "API is up and running!"
    
    return {"message": msg}

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    """
    Endpoint to predict feature embeddings from an uploaded image.
    Receives an image, processes it, and returns its feature embeddings.
    """
    try:        
        # Process the uploaded image
        processed_image = process_image(image.file)  # Convert the image to a tensor
        processed_image = processed_image.to(torch.device('cpu'))  # Ensure it's on CPU

        # Extract features (embeddings) using the model
        features = model.predict(processed_image)
        features = features.cpu().numpy().tolist()

        return JSONResponse(content={
            "features": features  # Return the image embeddings here
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    """
    Endpoint to find the top 5 most similar images to the uploaded image.
    Receives an image, processes it, and searches for similar images using the FAISS index.
    """
    print(text)
    try:        
        # Process the uploaded image
        processed_image = process_image(image.file)  # Convert the image to a tensor
        processed_image = processed_image.to(torch.device('cpu'))  # Ensure it's on CPU

        # # Verify the shape
        # print(f"Processed image shape: {processed_image.shape}") 
    
        # Extract features (embeddings) using the model
        query_embedding = model.predict(processed_image).cpu().numpy()

        # # Print the shape of the query embedding after model prediction
        # print(f"Query embedding shape after model prediction: {query_embedding.shape}")
        
        # Ensure the query embedding has the correct shape (1, 1000)
        if query_embedding.ndim == 3 and query_embedding.shape[1] == 1:
            query_embedding = np.squeeze(query_embedding, axis=1)
        # Now the query_embedding should have the shape (1, 1000)
        
        # print(f"Query embedding shape before FAISS: {query_embedding.shape}")
        
        # Perform the similarity search using FAISS
        distances, indices = faiss_index.search(query_embedding.astype('float32'), 5)  # Return top 5 similar images
        
        # Retrieve the IDs of the similar images
        similar_image_ids = [image_ids[i] for i in indices[0]]

        return JSONResponse(content={
            "similar_index": similar_image_ids,  # Return the index of similar images here
            "distances": distances[0].tolist()
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    
if __name__ == '__main__':
    print("Starting Uvicorn server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8080)
