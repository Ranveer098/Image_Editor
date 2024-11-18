import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io
import base64

# Set page config
st.set_page_config(page_title="Advanced Image Editor", page_icon="🖌️", layout="centered")

# Title
st.title("🖌️ Advanced Image Editor")

# File upload
uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    modified_image = original_image.copy()

    st.image(original_image, caption="Original Image", use_container_width=True)

    # Helper functions
    def pil_to_np(image):
        return np.array(image)

    def np_to_pil(image_np):
        return Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))

    # Image Processing Functions
    def crop_image(image, top, bottom, left, right):
        img_np = pil_to_np(image)
        cropped = img_np[top:bottom, left:right]
        return np_to_pil(cropped)

    def apply_negative(image):
        img_np = pil_to_np(image)
        negative = 255 - img_np
        return np_to_pil(negative)

    def detect_edges(image):
        img_np = pil_to_np(image).astype(np.float32)
        gray = np.dot(img_np[..., :3], [0.2989, 0.587, 0.114])
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = sobel_x.T
        edges_x = np.abs(np.convolve(gray.flatten(), sobel_x.flatten(), mode="same").reshape(gray.shape))
        edges_y = np.abs(np.convolve(gray.flatten(), sobel_y.flatten(), mode="same").reshape(gray.shape))
        edges = np.sqrt(edges_x**2 + edges_y**2)
        edges = (edges / edges.max()) * 255
        return Image.fromarray(edges.astype(np.uint8))

    def remove_background(image):
        img_np = pil_to_np(image)
        # Simplified k-means-based color clustering
        pixels = img_np.reshape(-1, 3)
        background_color = pixels[np.argmax(np.bincount(pixels[:, 0]))]  # Mode color
        mask = np.all(np.abs(img_np - background_color) < 30, axis=-1)  # Threshold tolerance
        img_np[mask] = [0, 0, 0]  # Replace background with black
        return np_to_pil(img_np)

    def blur_image(image, blur_radius):
        img_np = pil_to_np(image)
        kernel = np.ones((blur_radius, blur_radius)) / (blur_radius**2)
        blurred = np.convolve(img_np.flatten(), kernel.flatten(), mode="same").reshape(img_np.shape)
        return np_to_pil(blurred)


    def sharpen_image(image):
        img_np = pil_to_np(image).astype(np.float32)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # Apply sharpening filter to each channel independently
        sharpened = np.zeros_like(img_np)
        for channel in range(3):  # For RGB channels
            sharpened[..., channel] = np.clip(
                np.convolve(img_np[..., channel].flatten(), kernel.flatten(), mode="same").reshape(
                    img_np[..., channel].shape),
                0,
                255,
            )

        return np_to_pil((sharpened+image+image+image)/4)


    def adjust_contrast(image, factor):
        img_np = pil_to_np(image).astype(np.float32)
        mean = img_np.mean(axis=(0, 1))
        contrasted = np.clip((img_np - mean) * factor + mean, 0, 255)
        return np_to_pil(contrasted)

    def apply_sepia(image):
        img_np = pil_to_np(image).astype(np.float32)
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        sepia_img = img_np @ sepia_filter.T
        return np_to_pil(sepia_img)

    def apply_vignette(image):
        img_np = pil_to_np(image)
        rows, cols, _ = img_np.shape
        x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        vignette_mask = np.sqrt(x**2 + y**2)
        vignette_mask = 1 - vignette_mask / np.max(vignette_mask)
        vignette_mask = np.clip(vignette_mask, 0, 1)
        vignette_img = img_np * vignette_mask[:, :, np.newaxis]
        return np_to_pil(vignette_img)

    # Sidebar Controls
    st.sidebar.header("Adjustments")
    crop_top = st.sidebar.slider("Crop Top", 0, original_image.height, 0)
    crop_bottom = st.sidebar.slider("Crop Bottom", 0, original_image.height, original_image.height)
    crop_left = st.sidebar.slider("Crop Left", 0, original_image.width, 0)
    crop_right = st.sidebar.slider("Crop Right", 0, original_image.width, original_image.width)
    apply_crop = st.sidebar.button("Apply Crop")

    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
    apply_contrast = st.sidebar.button("Adjust Contrast")

    blur_radius = st.sidebar.slider("Blur Radius", 1, 15, 1)
    apply_blur_effect = st.sidebar.button("Apply Blur")

    st.sidebar.header("Effects")
    negative = st.sidebar.button("Negative")
    edges = st.sidebar.button("Detect Edges")
    sharpen = st.sidebar.button("Sharpen")
    remove_bg = st.sidebar.button("Remove Background")
    sepia = st.sidebar.button("Apply Sepia")
    vignette = st.sidebar.button("Apply Vignette")
    reset = st.sidebar.button("Reset")

    # Apply transformations
    if apply_crop:
        modified_image = crop_image(modified_image, crop_top, crop_bottom, crop_left, crop_right)

    if apply_contrast:
        modified_image = adjust_contrast(modified_image, contrast)

    if apply_blur_effect:
        modified_image = blur_image(modified_image, blur_radius)

    if negative:
        modified_image = apply_negative(modified_image)

    if edges:
        modified_image = detect_edges(modified_image)

    if sharpen:
        modified_image = sharpen_image(modified_image)

    if remove_bg:
        modified_image = remove_background(modified_image)

    if sepia:
        modified_image = apply_sepia(modified_image)

    if vignette:
        modified_image = apply_vignette(modified_image)

    if reset:
        modified_image = original_image.copy()

    # Display modified image
    st.image(modified_image, caption="Modified Image", use_container_width=True)

    # Download button
    buffer = io.BytesIO()
    modified_image.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    st.markdown(f"""
    <a href="data:image/png;base64,{b64}" download="modified_image.png">
        <button>Download Image</button>
    </a>
    """, unsafe_allow_html=True)
else:
    st.info("Upload an image to begin editing!")
