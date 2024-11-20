import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
# Inject CSS to hide the GitHub "Fork" button
hide_fork_button = """
<style>
header a[title="View source"], header a[aria-label="View source"] {
    display: none !important;
}
</style>
"""

st.markdown(hide_fork_button, unsafe_allow_html=True)
# Hide Streamlit menu and add footer
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        footer:after {
            content:'This app is in its early stage. We recommend you to seek professional advice from a dermatologist. Thank you.'; 
            visibility: visible;
            display: block;
            position: relative;
            padding: 5px;
            top: 2px;
        }
        .center { text-align: center; }
    </style>
    """, 
    unsafe_allow_html=True
)

# Load the pre-trained model
model = tf.keras.models.load_model('best_model.h5')

# Define labels for categories
labels = {
    0: 'Chickenpox',
    1: 'Cowpox',
    2: 'HFMD',
    3: 'Healthy',
    4: 'Measles',
    5: 'MPOX'
}

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to 224x224
    image_array = np.expand_dims(np.array(image), axis=0)  # Convert and expand dimensions
    return image_array

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    label_index = np.argmax(prediction)
    predicted_label = labels[label_index]
    confidence = prediction[0][label_index] * 100
    return predicted_label, confidence

# Streamlit app
def main():
    st.markdown("<h1 class='center'>MPOX Skin Lesion Classifier</h1>", unsafe_allow_html=True)

    # Image upload options
    source = st.radio('Pick one', ['Upload from gallery', 'Capture by camera'])
    uploaded_file = st.camera_input("Take a picture") if source == 'Capture by camera' else st.file_uploader("Choose an image", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Process and classify the image
        predicted_label, confidence = predict(image)
        
        # Display results
        st.markdown(f"<h3 class='center'>This might be:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 class='center'>{predicted_label}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='center'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
if __name__ == '__main__':
    main()
