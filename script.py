import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import io

st.title("📐 Architectural Plan Difference Viewer")

uploaded_file1 = st.file_uploader("Upload First PDF (Original Plan)", type="pdf")
uploaded_file2 = st.file_uploader("Upload Second PDF (Updated Plan)", type="pdf")

def pdf_to_image(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)                     # first page
    pix = page.get_pixmap(dpi=150)              # render at 150 DPI
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return np.array(img)

if uploaded_file1 and uploaded_file2:
    try:
        # Convert PDFs to RGB images (first page only)
        img1_rgb = pdf_to_image(uploaded_file1.read())
        img2_rgb = pdf_to_image(uploaded_file2.read())

        # Convert to grayscale
        img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Align images using ECC
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        _, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        img2 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))

        # Detect differences
        diff = cv2.absdiff(img1, img2)
        _, mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        if not np.any(mask):
            st.success("✅ No differences found between the two plans.")
        else:
            # Prepare overlay
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            img2_color = (img2_color * 0.1).astype(np.uint8)  # darken
            overlay = img2_color.copy()
            overlay[mask > 0] = [255, 0, 0]                    # red highlights

            # Show & download
            result = Image.fromarray(overlay)
            buf = io.BytesIO()
            result.save(buf, format="PNG")

            st.success("⚠️ Differences detected. Click below to download the overlay:")
            st.image(result, caption="🔍 Differences Highlighted in Bright Red", use_container_width=True)
            st.download_button(
                label="📥 Download Difference Overlay",
                data=buf.getvalue(),
                file_name="difference_overlay.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
