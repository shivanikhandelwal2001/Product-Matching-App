import streamlit as st
import numpy as np
import requests
import base64
from PIL import Image
from io import BytesIO
from qdrant_client import QdrantClient, models
from pymongo import MongoClient
from transformers import AutoProcessor, BitsAndBytesConfig
import torch


@st.cache_resource
def get_client():
    q_client = QdrantClient(
        url=st.secrets["qdrant_db_url"],
        api_key=st.secrets["qdrant_api_key"]
    )

    username = st.secrets["mongo_initdb_root_username"]
    password = st.secrets["mongo_initdb_root_password"]
    mongo_client = MongoClient(f"mongodb://{username}:{password}@localhost:27017")
    db = mongo_client["product_db"]
    products_collection = db["products"]
    logs_collection = db["logs"]

    return q_client, products_collection, logs_collection


# Function to generate embeddings
def generate_embeddings(image):
    """Extracts image and text embeddings from the uploaded image."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    image_embedding = model.encode_image(image).cpu().detach().numpy().flatten().tolist()
    text_embedding = model.encode_text(caption).cpu().detach().numpy().flatten().tolist()
    return image_embedding, text_embedding, caption

# Function to search in Qdrant
def search_similar_images(image_embedding, text_embedding, top_k=6):
    """Performs nearest neighbor search in Qdrant."""
    query_embedding = np.concatenate((image_embedding, text_embedding)).tolist()
    results = qdrant.search(
        collection_name="image_text_vectors",
        query_vector=query_embedding,
        limit=top_k
    )
    return results

# Function to fetch metadata from MongoDB
def fetch_metadata(image_ids):
    """Fetches metadata for retrieved images from MongoDB."""
    return list(products_collection.find({"image_id": {"$in": image_ids}}))

# Function to log queries
def log_query(query_image, results):
    """Stores user queries, results, and execution logs in MongoDB."""
    logs_collection.insert_one({"query_image": query_image, "results": results})


# Streamlit UI
st.title("🔍 Product Matching System")
st.write("Upload an image to find similar products!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Extracting embeddings and searching..."):
        image_embedding, text_embedding, caption = generate_embeddings(image)
        results = search_similar_images(image_embedding, text_embedding)
        
        image_ids = [res.id for res in results]
        metadata = fetch_metadata(image_ids)
        
        log_query(uploaded_file.name, metadata)
    
    st.subheader("🔹 Top 6 Similar Images")
    cols = st.columns(3)
    
    for idx, data in enumerate(metadata):
        with cols[idx % 3]:
            img_data = requests.get(data["image_url"]).content
            st.image(Image.open(BytesIO(img_data)), caption=data["caption"], use_column_width=True)

