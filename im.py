import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import cv2
import numpy as np
import io
import base64
from rembg import remove  # For background removal

# Custom CSS for styling
st.markdown("""
    <style>
        /* Background styling */
        body {
            background: linear-gradient(120deg, #ff9a9e 0%, #fad0c4 100%);
            color: white;
        }
        
        /* Main title styling */
        .main-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
            margin-top: -20px;
            margin-bottom: 30px;
            animation: fadeIn 2s;
        }

        /* Button styling */
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px;
            font-size: 1em;
            width: 140px;
            transition: background-color 0.3s ease, transform 0.2s ease;
            margin: 8px;
            display: inline-block;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: scale(1.05);
            cursor: pointer;
        }

        /* Fade-in animation for title */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Download button styling */
        .download-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .download-button button {
            background-color: #FF5722;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-size: 1.2em;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .download-button button:hover {
            background-color: #E64A19;
            transform: scale(1.05);
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>Image Editor Tool</h1>", unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)  # Store the original image
    modified_image = original_image.copy()  # Initialize modified image

    # Display the original image at a smaller size
    st.image(original_image, caption="Uploaded Image", width=300)

    # Resize Function
    def resize_image(image, width, height):
        return image.resize((width, height))

    # Brightness Adjustment
    def adjust_brightness(image, brightness_value):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness_value)

    # Saturation Adjustment
    def adjust_saturation(image, saturation_value):
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(saturation_value)

    # Apply Sharpen
    def apply_sharpen(image):
        return image.filter(ImageFilter.SHARPEN)

    # Edge Detection
    def detect_edges(image):
        image_cv = np.array(image.convert("L"))
        edges = cv2.Canny(image_cv, 100, 200)
        return Image.fromarray(edges)

    # Upscale and Sharpen
    def upscale_and_sharpen(image, scale_factor=2):
        image_np = np.array(image)
        upscaled_image = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        upscaled_pil_image = Image.fromarray(upscaled_image).filter(ImageFilter.SHARPEN)
        return upscaled_pil_image

    # Background Removal
    def remove_background(image):
        image_np = np.array(image)  # Convert PIL Image to numpy array
        no_bg = remove(image_np)  # Remove background
        return Image.fromarray(no_bg)  # Convert back to PIL Image

    # Text Watermark (Tiled across the image)
    def add_watermark(image, watermark_text):
        watermarked_image = image.copy()
        draw = ImageDraw.Draw(watermarked_image)
        width, height = watermarked_image.size
        for x in range(0, width, 100):
            for y in range(0, height, 100):
                draw.text((x, y), watermark_text, fill="white")
        return watermarked_image

    # Interactive Controls
    width = st.slider("Width", 100, 1000, original_image.width)
    height = st.slider("Height", 100, 1000, original_image.height)
    brightness_value = st.slider("Brightness", 0.1, 2.0, 1.0)
    saturation_value = st.slider("Saturation", 0.1, 2.0, 1.0)
    watermark_text = st.text_input("Enter watermark text")

    # Button Layout in a Horizontal Line
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("Apply"):
            modified_image = resize_image(modified_image, width, height)
            modified_image = adjust_brightness(modified_image, brightness_value)
            modified_image = adjust_saturation(modified_image, saturation_value)
            modified_image = apply_sharpen(modified_image)
            if watermark_text:
                modified_image = add_watermark(modified_image, watermark_text)
            st.image(modified_image, caption="Modified Image", width=500)
    with col2:
        if st.button("Grayscale"):
            modified_image = ImageOps.grayscale(modified_image)
            st.image(modified_image, caption="Grayscale Image", width=500)
    with col3:
        if st.button("Edges"):
            modified_image = detect_edges(modified_image)
            st.image(modified_image, caption="Edge Detected Image", width=500)
    with col4:
        if st.button("Upscale"):
            modified_image = upscale_and_sharpen(modified_image)
            st.image(modified_image, caption="Upscaled and Sharpened Image", width=500)
    with col5:
        if st.button("Remove BG"):
            modified_image = remove_background(modified_image)
            st.image(modified_image, caption="Image with Background Removed", width=500)
    with col6:
        if st.button("Clear All"):
            modified_image = original_image.copy()
            st.image(modified_image, caption="Original Image", width=500)

    # Convert modified image to bytes for download
    buffer = io.BytesIO()
    modified_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode the image to base64
    b64 = base64.b64encode(buffer.getvalue()).decode()

    # Styled Download Button
    st.markdown(f"""
    <div class="download-button">
        <a href="data:image/png;base64,{b64}" download="modified_image.png">
            <button>Download Image</button>
        </a>
    </div>
    """, unsafe_allow_html=True)

else:
    st.write("Please upload an image to apply transformations.")
