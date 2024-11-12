import streamlit as st 
from PIL import Image, ImageDraw
import cv2
import numpy as np
import io
import base64
from rembg import remove  # Background removal

# Set the page title and icon
st.set_page_config(page_title="Image Editor Tool", page_icon="🖌️")

# Custom CSS for styling and animations
st.markdown("""
    <style>
        /* Animated Border for the file uploader */
        .file-upload {
            border: 2px dashed #6C63FF;
            border-radius: 8px;
            padding: 16px;
            animation: glow 2s ease-in-out infinite;
        }
        
        /* Glowing effect */
        @keyframes glow {
            0% { border-color: #6C63FF; box-shadow: 0 0 10px #6C63FF; }
            50% { border-color: #FFD700; box-shadow: 0 0 20px #FFD700; }
            100% { border-color: #6C63FF; box-shadow: 0 0 10px #6C63FF; }
        }

        /* Main title styling with animation */
        .main-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
            margin-top: -20px;
            margin-bottom: 30px;
            animation: color-change 3s infinite;
        }

        /* Title color-changing animation */
        @keyframes color-change {
            0% { color: #4CAF50; }
            50% { color: #FFD700; }
            100% { color: #4CAF50; }
        }

        /* Custom button colors and hover effects */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-size: 1em;
            width: 120px;
            margin: 4px 2px; /* Added margin to space out buttons */
            transition: background-color 0.3s ease, transform 0.2s ease;
            display: inline-block;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
            cursor: pointer;
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

# Title with animation
st.markdown("<h1 class='main-title'>Image Editor Tool</h1>", unsafe_allow_html=True)

# File uploader with animated border
st.markdown("<div class='file-upload'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    modified_image = original_image.copy()

    # Display the original image
    st.image(original_image, caption="Uploaded Image", width=300)

    # Raw implementation of image processing functions

    # Resize Function (Raw implementation)
    def resize_image(image, width, height):
        image_np = np.array(image)
        resized_image = cv2.resize(image_np, (width, height), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(resized_image)

    # Brightness Adjustment (Raw implementation)
    def adjust_brightness(image, brightness_value):
        image_np = np.array(image, dtype=np.float32)
        image_np = np.clip(image_np * brightness_value, 0, 255)
        return Image.fromarray(image_np.astype(np.uint8))

    # Saturation Adjustment (Raw implementation)
    def adjust_saturation(image, saturation_value):
        image_np = np.array(image.convert("RGB"), dtype=np.float32)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray_3d = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
        image_np = np.clip((image_np * saturation_value) + (gray_3d * (1 - saturation_value)), 0, 255)
        return Image.fromarray(image_np.astype(np.uint8))

    # Sharpening (Raw implementation)
    def apply_sharpen(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image_np = np.array(image)
        sharpened = cv2.filter2D(image_np, -1, kernel)
        return Image.fromarray(sharpened)

    # Edge Detection (Raw implementation)
    def detect_edges(image):
        image_np = np.array(image.convert("L"))
        edges = cv2.Canny(image_np, 100, 200)
        return Image.fromarray(edges)

    # Background Removal
    def remove_background(image):
        image_np = np.array(image)  # Convert PIL Image to numpy array
        no_bg = remove(image_np)  # Remove background using rembg
        return Image.fromarray(no_bg)  # Convert back to PIL Image

    # Upscaling (Raw implementation)
    def upscale_image(image, scale_factor=2):
        image_np = np.array(image)
        upscaled_image = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(upscaled_image)

    # Add Text Watermark
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
    cols = st.columns(7)  # Spread buttons into 7 evenly spaced columns
    with cols[0]:
        if st.button("Apply"):
            modified_image = resize_image(modified_image, width, height)
            modified_image = adjust_brightness(modified_image, brightness_value)
            modified_image = adjust_saturation(modified_image, saturation_value)
            modified_image = apply_sharpen(modified_image)
            if watermark_text:
                modified_image = add_watermark(modified_image, watermark_text)
            st.image(modified_image, caption="Modified Image", width=500)
    with cols[1]:
        if st.button("Grayscale"):
            modified_image = modified_image.convert("L")
            st.image(modified_image, caption="Grayscale Image", width=500)
    with cols[2]:
        if st.button("Edges"):
            modified_image = detect_edges(modified_image)
            st.image(modified_image, caption="Edge Detected Image", width=500)
    with cols[3]:
        if st.button("Sharpen"):
            modified_image = apply_sharpen(modified_image)
            st.image(modified_image, caption="Sharpened Image", width=500)
    with cols[4]:
        if st.button("Remove BG"):
            modified_image = remove_background(modified_image)
            st.image(modified_image, caption="Image with Background Removed", width=500)
    with cols[5]:
        if st.button("Upscale"):
            modified_image = upscale_image(modified_image)
            st.image(modified_image, caption="Upscaled Image", width=500)
    with cols[6]:
        if st.button("Clear All"):
            modified_image = original_image.copy()
            st.image(modified_image, caption="Original Image", width=500)

    # Convert modified image to bytes for download
    buffer = io.BytesIO()
    modified_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode the image to base64 for download link
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
