import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Define the model predictions functions
@st.cache_resource  # Cache the model loading to optimize performance
def load_model():
    # Load the pre-trained model from the specified file
    model = tf.keras.models.load_model("potato_disease_model.keras")
    print("Model input shape:", model.input_shape)  # Debug line (check terminal output)
    return model

def model_prediction(test_image):
    """Function to load the trained model and predict the class of the uploaded image."""
    model = load_model()  # model load using the cached function
    
    # Convert the PIL image to a file-like object (BytesIO)
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Load and resize image to 256x256 (MUST match training size)
    image = tf.keras.preprocessing.image.load_img(img_byte_arr, target_size=(256, 256))
    
    # Convert to numpy array and expand dimensions
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert to batch (batch size = 1)
    
    # Predict the class
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of highest probability class


# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Detection"])

# Home Page
if app_mode == "Home":
    st.header("Potato Leaf Disease Detection System")
    st.image("Potato Image.jpeg", use_column_width=True)
    st.markdown("""
    ## Welcome to the Potato Leaf Disease Detection System! 🥔🌿
    This system helps detect potato leaf diseases using AI.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    This app detects Early Blight, Late Blight, and Healthy potato leaves.
    """)

# Disease Detection Page
elif app_mode == "Disease Detection":
    st.title("Disease Detection")
    uploaded_file = st.file_uploader("Upload potato leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.session_state.image = Image.open(uploaded_file)
        st.image(st.session_state.image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            st.write("Analyzing...")
            try:
                result_index = model_prediction(st.session_state.image)
                class_names = ['Early Blight', 'Late Blight', 'Healthy']
                prediction = class_names[result_index]
                st.success(f"Prediction: {prediction}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.write("Please ensure:")
                st.write("- Image shows a single potato leaf")
                st.write("- File is JPG/PNG format")
                st.write("- Model file exists (potato_disease_model.keras)")