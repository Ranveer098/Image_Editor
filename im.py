import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import cv2
import numpy as np

st.title("Advanced Image Manipulation and Image Editor Tool")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

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

    # Resize controls
    width = st.slider("Width", 100, 1000, image.width)
    height = st.slider("Height", 100, 1000, image.height)
    if st.button("Resize Image"):
        resized_image = resize_image(image, width, height)
        st.image(resized_image, caption="Resized Image", use_column_width=True)

    # Brightness control
    brightness_value = st.slider("Brightness", 0.1, 2.0, 1.0)
    if st.button("Adjust Brightness"):
        bright_image = adjust_brightness(image, brightness_value)
        st.image(bright_image, caption="Brightness Adjusted Image", use_column_width=True)

    # Saturation control
    saturation_value = st.slider("Saturation", 0.1, 2.0, 1.0)
    if st.button("Adjust Saturation"):
        saturated_image = adjust_saturation(image, saturation_value)
        st.image(saturated_image, caption="Saturation Adjusted Image", use_column_width=True)

    # Sharpen
    if st.button("Sharpen Image"):
        sharpened_image = apply_sharpen(image)
        st.image(sharpened_image, caption="Sharpened Image", use_column_width=True)

    # Grayscale Conversion
    if st.button("Convert to Grayscale"):
        grayscale_image = ImageOps.grayscale(image)
        st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

    # Edge Detection
    if st.button("Detect Edges"):
        edge_image = detect_edges(image)
        st.image(edge_image, caption="Edge Detected Image", use_column_width=True)

    # Upscale and Sharpen
    if st.button("Upscale and Sharpen"):
        upscaled_image = upscale_and_sharpen(image)
        st.image(upscaled_image, caption="Upscaled and Sharpened Image", use_column_width=True)

    # Text Watermark
    watermark_text = st.text_input("Enter watermark text")
    if watermark_text and st.button("Apply Text Watermark"):
        watermarked_image = image.copy()
        draw = ImageDraw.Draw(watermarked_image)
        draw.text((10, 10), watermark_text, fill="white")
        st.image(watermarked_image, caption="Watermarked Image", use_column_width=True)
else:
    st.write("Please upload an image to apply transformations.")
