import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
from pathlib import Path
import io

st.set_page_config(page_title="Retail Product Detection", page_icon="🛍️", layout="wide")

st.title("🛍️ Retail Product Detection")
st.write("Detect objects in a retail image using the pre-trained YOLO11 nano model.")

@st.cache_resource
def load_model():
    # Ultralytics downloads yolo11n.pt on first run if it is not present.
    return YOLO("yolo11n.pt")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "webp"]
)
confidence = st.slider("Confidence threshold", min_value=0.05, max_value=0.95,
                       value=0.25, step=0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    if st.button("Detect objects", type="primary"):
        with st.spinner("Loading model and detecting objects..."):
            model = load_model()
            results = model.predict(source=image, conf=confidence, imgsz=640, verbose=False)
            result = results[0]
            annotated = Image.fromarray(result.plot()[:, :, ::-1])

        with right:
            st.subheader("Detection result")
            st.image(annotated, use_container_width=True)

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            names = [model.names[int(class_id)] for class_id in class_ids]
            counts = Counter(names)

            st.subheader("Detection summary")
            c1, c2 = st.columns(2)
            c1.metric("Total detections", len(names))
            c2.metric("Unique classes", len(counts))

            st.write("**Objects detected**")
            st.table([{"Object": name, "Count": count}
                      for name, count in counts.items()])

            st.write("**Individual detections**")
            st.dataframe(
                [{"Object": name, "Confidence": f"{score:.1%}"}
                 for name, score in zip(names, confidences)],
                use_container_width=True,
                hide_index=True
            )
        else:
            with right:
                st.info("No objects detected. Try another image or lower the confidence threshold.")

        output = io.BytesIO()
        annotated.save(output, format="JPEG")
        st.download_button(
            "Download detection image",
            data=output.getvalue(),
            file_name="retail_detection_result.jpg",
            mime="image/jpeg"
        )

st.caption("Note: YOLO11n is a general-purpose object detector trained on COCO classes. It may not recognize every retail product or distinguish fresh from rotten produce without a suitable trained model.")
