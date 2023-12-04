# Import necessary libraries
from roboflow import Roboflow
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Abhinaya-Detection",
    page_icon="📚",
)
# Authenticate with Roboflow
rf = Roboflow(api_key="D7H5KMdYl4UF9q0Q1Z1C")
project = rf.workspace().project("parking-space-ipm1b")
model = project.version(4).model

# Streamlit app
st.title("Detection")

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

    # Display the result
    st.write("Prediction Result:")

    # visualize your prediction
    prediction_image = model.predict("uploaded_image.jpg", confidence=confidence_threshold, overlap=overlap_threshold)
    prediction_image.save("prediction.jpg")
    st.image("prediction.jpg")

    # Display the threshold values
    st.write(f"Confidence Threshold used for prediction: {confidence_threshold}")
    st.write(f"Overlap Threshold used for prediction: {overlap_threshold}")

    # Extract and display the count of each class
    class_counts = {}
    if "objects" in result:
        for obj in result["objects"]:
            class_name = obj["class"]
            confidence = obj["confidence"]
            if confidence >= confidence_threshold:
                # Increment the count for the class
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Display the counts
    st.write("Car Counts for Each Class:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name}: {count}")
