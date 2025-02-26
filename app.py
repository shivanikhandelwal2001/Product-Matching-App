from qdrant_client import QdrantClient
import streamlit as st
import base64
from io import BytesIO
import os


collection_name = "product_matching"
st.title("Product Matching")


@st.cache_resource
def get_client():
    q_client = QdrantClient(
        url=st.secrets["qdrant_db_url"],
        api_key=st.secrets["qdrant_api_key"]
    )
    return q_client


if "selected_product" not in st.session_state:
    st.session_state.selected_product = None


def set_selected_product(product):
    st.session_state.selected_product = product


def get_initial_products():
    client = get_client()
    products, _ = client.scroll(collection_name=collection_name, with_vectors=False, limit=12)

    return products


def get_similar_products():
    client = get_client()
    if st.session_state.selected_product is not None:
        products = client.recommend(collection_name=collection_name,
                                    positive=[st.session_state.selected_product.id], 
                                    limit=6)
        
    return products
    

def get_different_products():
    client = get_client()
    if st.session_state.selected_product is not None:
        products = client.recommend(collection_name=collection_name,
                                    positive=[120],
                                    negative=[st.session_state.selected_product.id], 
                                    limit=6)

    return products


def get_bytes_image(base64_image):
    return BytesIO(base64.b64decode(base64_image))


if st.session_state.selected_product:
    similar_products = get_similar_products()
    different_products = get_different_products()
else:
    products = get_initial_products()


if st.session_state.selected_product:
    image = get_bytes_image(st.session_state.selected_product.payload["base64"])
    st.header("Selected Product")
    st.image(image)
    st.write(st.session_state.selected_product.payload["label"])
    st.divider()


if st.session_state.selected_product:
    s_columns = st.columns(3)
    for idx, product in enumerate(similar_products):
        col_idx = idx % 3
        image = get_bytes_image(product.payload["base64"])
        with s_columns[col_idx]:
            st.write(product.payload["label"])
            st.image(image)

    st.divider()
    st.header("Different Products")

    d_columns = st.columns(3)
    for idx, product in enumerate(different_products):
        col_idx = idx % 3
        image = get_bytes_image(product.payload["base64"])
        with d_columns[col_idx]:
            st.write(product.payload["label"])
            st.image(image)
            st.button("Find Similar Products", key=product.id, on_click=set_selected_product, args=(product,))

else:
    columns = st.columns(3)
    for idx, product in enumerate(products):
        col_idx = idx % 3
        image = get_bytes_image(product.payload["base64"])
        with columns[col_idx]:
            st.write(product.payload["label"])
            st.image(image)
            st.button("Find Similar Products", key=product.id, on_click=set_selected_product, args=(product,))
