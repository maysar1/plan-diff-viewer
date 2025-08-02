import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import io

st.title("📐 Architectural Plan Difference Viewer")

# Upload PDF files
uploaded_file1 = st.file_uploader("Upload First PDF (Original Plan)", type="pdf")
uploaded_file2 = st.file_uploader("Upload Second PDF (Updated Plan)", type="pdf")

def pdf_to_image(pdf_bytes):
    """Convert PDF bytes to an RGB numpy array using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return np.array(img)

if uploaded_file1 and uploaded_file2:
    try:
        # Render PDFs to RGB images
        img1_rgb = pdf_to_image(uploaded_file1.read())
        img2_rgb = pdf_to_image(uploaded_file2.read())

        # Convert to grayscale for alignment and diff
        img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

        # Align grayscale images using ECC
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        _, warp_matrix = cv2.findTransformECC(img1_gray, img2_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        img2_gray_aligned = cv2.warpAffine(img2_gray, warp_matrix, (img1_gray.shape[1], img1_gray.shape[0]))
        img2_rgb_aligned = cv2.warpAffine(img2_rgb, warp_matrix, (img1_rgb.shape[1], img1_rgb.shape[0]))

        # Compute difference mask
        diff = cv2.absdiff(img1_gray, img2_gray_aligned)
        _, mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        # Prepare overlay with transparency
        base_img = img1_rgb.copy()
        highlight = base_img.copy()
        highlight[mask > 0] = [255, 0, 0]  # red highlights
        alpha = 0.6
        overlay = cv2.addWeighted(base_img, 1 - alpha, highlight, alpha, 0)

        # View mode selector
        view = st.radio("Choose view mode:", ["Original Plan", "Updated Plan", "Difference Overlay"])
        if view == "Original Plan":
            st.image(img1_rgb, caption="Original Plan", use_container_width=True)
        elif view == "Updated Plan":
            st.image(img2_rgb_aligned, caption="Updated Plan (aligned)", use_container_width=True)
        else:
            st.image(overlay, caption="🔍 Differences Highlighted", use_container_width=True)
            # Download difference overlay
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            st.download_button(
                label="📥 Download Difference Overlay",
                data=buf.getvalue(),
                file_name="difference_overlay.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
