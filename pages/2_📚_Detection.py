# Import necessary libraries
from roboflow import Roboflow
import streamlit as st
from PIL import Image
from PIL import ImageDraw

st.set_page_config(
    page_title="Abhinaya-Detection",
    page_icon="📚",
)

# Authenticate with Roboflow
rf = Roboflow(api_key="D7H5KMdYl4UF9q0Q1Z1C")
project = rf.workspace().project("parking-space-ipm1b")
model = project.version(4).model

# Streamlit app
st.title("Parking Detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Check if file is uploaded
if uploaded_file is not None:
    # Save the uploaded image as a local file
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Display the uploaded image
    st.image("uploaded_image.jpg", caption="Uploaded Image.", use_column_width=True)

    # Set default threshold
    confidence_threshold = st.slider("Confidence Threshold", min_value=0, max_value=100, value=50, step=5)
    overlap_threshold = st.slider("Overlap Threshold", min_value=0, max_value=100, value=50, step=5)

    # Infer on the uploaded image
    result = model.predict("uploaded_image.jpg", confidence=confidence_threshold, overlap=overlap_threshold).json()

    # Visualize your prediction
    prediction_image = model.predict("uploaded_image.jpg", confidence=confidence_threshold, overlap=overlap_threshold)
    prediction_image.save("prediction.jpg")
    st.image("prediction.jpg")

    # Display the threshold values
    st.write(f"Confidence Threshold used for prediction: {confidence_threshold}")
    st.write(f"Overlap Threshold used for prediction: {overlap_threshold}")

    # Make a prediction on another image
    image_path = "uploaded_image.jpg"  # Replace with the actual path
    prediction_result_another = model.predict(image_path, confidence=confidence_threshold, overlap=overlap_threshold).json()

    # Make predictions on another image with varying 'n'
    num_predictions = len(prediction_result_another['predictions'])

    # Assuming the structure is similar to the one you provided
    num_predictions_another = len(prediction_result_another.get('predictions', []))
    class_names_another = []

    # Iterate over all predictions
    for n in range(num_predictions_another):
        class_name = prediction_result_another['predictions'][n]['class']
        class_names_another.append(class_name)

    occupied_count = class_names_another.count("occupied")
    empty_count = class_names_another.count("empty")

    # Display the count using st.write
    st.write(f"Jumlah Mobil : {occupied_count}")
    st.write(f"Jumlah Lahan yang tersedia : {empty_count}")