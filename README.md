# Bollywood Celebrity Predictor

Upload a photo — the app detects your face and tells you which Bollywood celebrity you look most like, with a confidence score.

---

## What It Does

Upload any photo containing a face. The app:
1. Detects the face using **MTCNN**
2. Extracts a facial embedding using **VGG-Face** (via DeepFace)
3. Compares against pre-computed embeddings for **100 Bollywood celebrities**
4. Returns the closest match and a confidence percentage

---

## How It Works

```
Input Photo → MTCNN Face Detection → VGG-Face Embedding (4096-dim)
    → GlobalMaxPool2D → Normalize → Cosine Similarity vs 100 Celebrities
    → Top Match + Confidence Score
```

**Embedding pipeline:**
- DeepFace extracts a 4096-dimensional VGG-Face embedding
- Reshaped to (2, 2, 1024) → GlobalMaxPool2D → 512-dim normalized vector
- Cosine similarity compared against pre-computed embeddings for all celebrities

**Face detection:**
- MTCNN detects and crops the face region
- Auto-resizes large images (>800px) before detection for reliability

---

## Tech Stack

| Component        | Tool                                      |
|------------------|-------------------------------------------|
| Language         | Python 3                                  |
| Face Detection   | MTCNN                                     |
| Face Embeddings  | VGG-Face via DeepFace                     |
| Similarity       | Cosine Similarity — `scikit-learn`        |
| Image Processing | OpenCV, Pillow                            |
| Deep Learning    | TensorFlow / Keras                        |
| App              | Streamlit                                 |

---

## Celebrities Included (100 total)

Aamir Khan, Aishwarya Rai, Ajay Devgn, Akshay Kumar, Alia Bhatt, Amitabh Bachchan, Anushka Sharma, Hrithik Roshan, Katrina Kaif, Priyanka Chopra, Ranbir Kapoor, Ranveer Singh, Salman Khan, Shahid Kapoor, Shah Rukh Khan, and 85 more.

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/huzefa10/bollywood-celebrity-predictor.git
cd bollywood-celebrity-predictor

# Install dependencies
pip install -r Requirements.txt

# Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, then upload any face photo.

---

## Project Structure

```
bollywood-celebrity-predictor/
├── app.py                  # Streamlit app — face detection, embedding, prediction
├── feature_extractor1.py   # Script to re-generate features.pkl from data/
├── training.ipynb          # Training notebook
├── features.pkl            # Pre-computed VGG-Face embeddings (100 celebrities)
├── filenames.pkl           # Image path index for all celebrities
├── Requirements.txt
├── data/                   # Celebrity reference images (100 celebrities)
├── sample/                 # Sample test images
└── uploads/                # Runtime folder for user-uploaded images
```

---

## Screenshot

*Demo screenshot — add after recording*

---

## Future Improvements

- Expand to 500+ celebrities
- Return top-3 matches instead of just the best
- Add face alignment before embedding extraction
- Try ArcFace or FaceNet embeddings for better accuracy
- Deploy on Streamlit Cloud or Hugging Face Spaces
