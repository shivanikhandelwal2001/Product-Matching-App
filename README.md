# 🔍 Product Matching System

This project implements a **Product Matching System** that allows users to upload an image and retrieve the most visually and textually similar product images from a database. The system leverages **Qdrant for vector search**, **MongoDB for metadata storage**, **Qwen2.5-VL for image captioning**, and **CLIP for text and image embeddings**. The application is deployed using **Streamlit and Docker**, and the dataset used is **COCO (Common Objects in Context)**. Additionally, **model quantization** has been applied to optimize performance.

---

## Features

- **Image & Text Embedding Extraction**: Uses **CLIP** and **Qwen2.5-VL** models to generate **image** and **text** embeddings.
- **Quantization for Optimization**: Reduces model size and improves inference efficiency using **TensorRT-based quantization**.
- **Vector Search with Qdrant**: Performs **nearest neighbor search** using Qdrant's **vector database**.
- **Metadata Storage with MongoDB**: Stores product details including **captions, base64 image data, and metadata**.
- **Streamlit UI**: Allows users to upload images and view top **6 similar product matches**.
- **Dockerized Deployment**: Runs the complete system in a **Docker container** for easy deployment.
- **COCO Dataset**: Uses **COCO dataset** for generating product embeddings and metadata.
- **Logging with MongoDB**: Stores logs of queries and retrieval results.

---

## Project Structure

```
📁 Product-Matching-App
│── 📝 README.md                    # Project Documentation
│── 📝 requirements.txt             # Python Dependencies
│── 📝 packages.txt                 # System Dependencies (for Streamlit Community)
│── 📝 docker-compose.yml           # Docker Compose File
│── 🐳 Dockerfile                   # Docker Configuration
│── 📜 product_matching_app.py      # Main Streamlit App
│── 📜 generate_embeddings.ipynb    # Script for Generating Embeddings & Quantization
│── 📂 product_images               # COCO Dataset & Images
```

---

## Installation & Setup

```bash
pip install -r requirements.txt
```

#### **Run the Streamlit App Locally**

```bash
streamlit run product_matching_app.py
```

---

#### **Build and Run the Docker Container**

```bash
docker build -t product-matching-app .
docker run -p 8501:8501 product-matching-app
```

```bash
docker-compose up --build -d
```

---

## Technologies Used

| Component         | Tool/Technology                        |
| ----------------- | -------------------------------------- |
| **Frontend**      | Streamlit                              |
| **Backend**       | Python                                 |
| **Vector Search** | Qdrant                                 |
| **Database**      | MongoDB Atlas                          |
| **Embeddings**    | CLIP, Qwen2.5-VL                       |
| **Quantization**  | TensorRT-based Model Optimization      |
| **Deployment**    | Docker, Streamlit Cloud                |
| **Dataset**       | COCO Dataset                           |

---

## Process

### **1️⃣ Image Upload**

- User uploads an **image** via the Streamlit UI.
- The image is converted to **base64** and stored.

### **2️⃣ Caption Generation**

- Uses **Qwen2.5-VL** model to **generate a caption** for the image.

### **3️⃣ Embedding Generation**

- Extracts **image embeddings** using **CLIP**.
- Extracts **text embeddings** for the caption.

### **4️⃣ Model Quantization**

- Optimizes the model with **TensorRT** to reduce size and improve inference speed.

### **5️⃣ Vector Search with Qdrant**

- Performs a **nearest neighbor search** on stored product embeddings.

### **6️⃣ Metadata Retrieval from MongoDB**

- Retrieves **product details** such as name, image, caption.

### **7️⃣ Results Displayed in Streamlit**

- Shows **top 6** similar product images with metadata.

---

## Demo
![Demo](demo1.mp4)

---

## 👩‍💻 Author
Developed by Shivani Khandelwal.

Connect with me here - [Profile](https://linktr.ee/shivanikhandelwal)

For questions or improvements, feel free to reach out at [shivanikhandelwal487@gmail.com](mailto:shivanikhandelwal487@gmail.com).

---
