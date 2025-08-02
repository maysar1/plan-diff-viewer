import streamlit as st
from pdf2image import convert_from_bytes
import numpy as np
import cv2
from PIL import Image
import io

# Set your Poppler path (update if needed)
POPPLER_PATH = r"C:\Program Files\Release-24.08.0-0\poppler-24.08.0\Library\bin"

st.title("📐 Architectural Plan Difference Viewer")

uploaded_file1 = st.file_uploader("Upload First PDF (Original Plan)", type="pdf")
uploaded_file2 = st.file_uploader("Upload Second PDF (Updated Plan)", type="pdf")

if uploaded_file1 and uploaded_file2:
    try:
        # Convert PDFs to grayscale images (first page only)
        images1 = convert_from_bytes(uploaded_file1.read(), dpi=150, poppler_path=POPPLER_PATH)
        images2 = convert_from_bytes(uploaded_file2.read(), dpi=150, poppler_path=POPPLER_PATH)

        img1 = cv2.cvtColor(np.array(images1[0]), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(np.array(images2[0]), cv2.COLOR_RGB2GRAY)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Align images using ECC
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        (cc, warp_matrix) = cv2.findTransformECC(img1, img2, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        img2 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))

        # Detect differences
        diff = cv2.absdiff(img1, img2)
        _, mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        if not np.any(mask):
            st.success("✅ No differences found between the two plans.")
        else:
            # Convert aligned grayscale to color for overlay
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            img2_color = (img2_color * 0.1).astype(np.uint8)  # reduce brightness by 50%
            overlay = img2_color.copy()

            # Highlight changes in bright red (BGR format)
            overlay[mask > 0] = [255, 0, 0]

            # Convert overlay to PNG
            result = Image.fromarray(overlay)
            buf = io.BytesIO()
            result.save(buf, format="PNG")

            st.success("⚠️ Differences detected. Click below to download the overlay:")
            st.image(result, caption="🔍 Differences Highlighted in Bright Red", use_column_width=True)
            st.download_button(
                label="📥 Download Difference Overlay",
                data=buf.getvalue(),
                file_name="difference_overlay.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
