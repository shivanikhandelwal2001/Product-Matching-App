from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from quanto import quantize, freeze
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Record
from qdrant_client.http import models
from pymongo import MongoClient
import os
import streamlit as st
import requests
import asyncio
import json
import time
import torch
import base64
from io import BytesIO
from glob import glob
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())


params = {
    "VISION_MODEL": "Qwen/Qwen2.5-VL-3B-Instruct",
    "EMBEDDING_MODEL": "openai/clip-vit-base-patch32",
    "BATCH_SIZE": 4,
    "IMAGE_NUM": 1000,
    "IMAGE_SIZE": (224, 224)
}


@st.cache_resource
def load_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        params["VISION_MODEL"],
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )

    qwen_processor = AutoProcessor.from_pretrained(params["VISION_MODEL"])


    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    clip_model = CLIPModel.from_pretrained(
        params["EMBEDDING_MODEL"],
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    ).to(device)

    clip_processor = CLIPProcessor.from_pretrained(params["EMBEDDING_MODEL"])
    clip_tokenizer = CLIPTokenizer.from_pretrained(params["EMBEDDING_MODEL"])

    return qwen_model, qwen_processor, clip_model, clip_processor, clip_tokenizer


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


def generate_captions(image):
    qwen_model, qwen_processor, _, _, _ = load_model()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in a few words."}
            ],
        }
    ]

    # Preparation for inference

    text = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    # Inference: Generation of the output
    with torch.inference_mode():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=64)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def get_text_embedding(text): 
    _, _, clip_model, clip_processor, clip_tokenizer = load_model()

    inputs = clip_tokenizer(text, return_tensors="pt").to(device)
    text_embeddings = clip_model.get_text_features(**inputs)
    with torch.inference_mode():
        embedding_as_np = text_embeddings.cpu().detach().numpy().flatten().tolist()

    return embedding_as_np


def get_image_embedding(image):
    _, _, clip_model, clip_processor, clip_tokenizer = load_model()

    image = clip_processor(text=None, images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode():
        embedding = clip_model.get_image_features(image)
    embedding_as_np = embedding.cpu().detach().numpy().flatten().tolist()

    return embedding_as_np


# Function to generate embeddings
def generate_embeddings(image):
    """Extracts image and text embeddings from the uploaded image."""
    caption = generate_captions(image)
    image_embedding = get_image_embedding(image)
    text_embedding = get_text_embedding(caption)
    
    return image_embedding, text_embedding, caption


# Function to search in Qdrant
def search_similar_images(image_embedding, text_embedding, top_k=6):
    """Performs nearest neighbor search in Qdrant."""
    q_client, _, _ = get_client()

    query_embedding = image_embedding
    results = q_client.search(
        collection_name="product_matching",
        query_vector=query_embedding,
        limit=top_k
    )

    return results


# Function to fetch metadata from MongoDB
def fetch_metadata(image_ids):
    """Fetches metadata for retrieved images from MongoDB."""
    _, products_collection, _ = get_client()

    return list(products_collection.find({"image_id": {"$in": image_ids}}))


# Function to log queries
def log_query(query_image, base64_image, results):
    """Stores user queries, results, and execution logs in MongoDB."""
    _, _, logs_collection = get_client()
    logs_collection.insert_one({"query_image": query_image, "base64": base64_image, "results": results})


def get_bytes_image(base64_image):
    """Converts a Base64 image to a BytesIO object."""
    return BytesIO(base64.b64decode(base64_image))


def image_to_base64(image):
    """Converts a PIL image to Base64 encoding."""
    buffered = BytesIO()
    image.save(buffered, format="jpeg")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return base64_image


# Streamlit UI
st.title("🔍 Product Matching System")
st.write("Upload an image to find similar products!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    base_image = image_to_base64(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Extracting embeddings and searching..."):
        image_embedding, text_embedding, caption = generate_embeddings(image)
        results = search_similar_images(image_embedding, text_embedding)
        
        image_ids = [res.id for res in results]
        metadata = fetch_metadata(image_ids)
        
        log_query(uploaded_file.name, base_image, metadata)
    
    st.subheader("Similar Images")
    cols = st.columns(3)
    
    for idx, data in enumerate(metadata):
        with cols[idx % 3]:
            image = get_bytes_image(data["base64"])
            st.image(image, caption=data["captions"], use_container_width=True)

