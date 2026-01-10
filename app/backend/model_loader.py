import streamlit as st
from ultralytics import YOLO
from app import model_store

@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    """
    This runs ONCE per Streamlit server start.
    Equivalent to FastAPI @app.on_event("startup")
    """
    
    shelf_detector_v14_path = "models/shelf_detector_v14/weights/best.pt"
    model_store.shelf_detector = YOLO(shelf_detector_v14_path)

    product_model_path = "models/product_recognition_yolo11/weights/best.pt"
    model_store.product_object_model = YOLO(product_model_path)

    product_rec_model_path = "models/rpc_yolov11_4dh3/weights/best.pt"
    model_store.product_rec_model = YOLO(product_rec_model_path)

    print("models loaded at startup")

    return True
