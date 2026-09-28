# Retail Product Detection — Streamlit

This app converts `Untitled10.ipynb` into a Streamlit application using the pre-trained YOLO11 nano model.

## Run locally

1. Install Python 3.10 or newer.
2. Extract this ZIP and open a terminal in the extracted folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

The first run downloads `yolo11n.pt`, so an internet connection is required. Upload a JPG, JPEG, PNG, or WEBP image, set the confidence threshold, and click **Detect objects**.

## Deploy on Streamlit Community Cloud

1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. In Streamlit Community Cloud, choose **Create app** and select that repository.
3. Set the main file path to `app.py`, then deploy.

The model is downloaded automatically by Ultralytics on first run. YOLO11n is a general-purpose COCO detector; it is not specifically trained to classify fruit as fresh or rotten.
