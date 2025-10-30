# Reviewer Recommender System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reviewer-recommender-app-kaash.streamlit.app/)

This project is a functional, NLP-powered system designed to recommend reviewers for academic research papers. Built as a part of an assignment, this app takes a research paper (in PDF format) as input and outputs a ranked list of the most suitable reviewers from a pre-built database of authors.

The live, deployed version of this app is available here:
**[https://reviewer-recommender-app-kaash.streamlit.app/](https://reviewer-recommender-app-kaash.streamlit.app/)**

---

## 🛡️ Core Features: The "True Fallback" Architecture

This system is built to be robust against the most common problem in document analysis: poorly formatted PDFs. It uses a **"True Fallback"** architecture to ensure that no relevant paper is missed, even if it lacks clear "Abstract" or "Introduction" headings.

It performs **two separate, parallel searches** for every query:

1.  **Search #1: Abstract-to-Abstract (Primary)**
    * The system first extracts the **"Best Text"** (Abstract + Introduction) from your uploaded paper.
    * It compares this "Best Text" against the "Best Text" database of all 600+ papers.
    * This is an "apples-to-apples" comparison that provides highly accurate scores for well-formatted papers.

2.  **Search #2: FullText-to-FullText (Fallback)**
    * The system also extracts the **"Full Text"** (cleaned of references) from your uploaded paper.
    * It compares this "Full Text" against the "Full Text" database.
    * This "oranges-to-oranges" search acts as a fallback, ensuring that if a relevant paper has no abstract, it is *still* found based on its full content.

The final score for any paper is the **`max()` of the two search results**, guaranteeing that the strongest possible connection is found.

---

## 🚀 How to Run This Project Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Kaashith25/Reviewer-Recommender-App.git](https://github.com/Kaashith25/Reviewer-Recommender-App.git)
    cd Reviewer-Recommender-App
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\Activate.ps1

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app_streamlit.py
    ```
    Your browser should automatically open to the app.

---

## ⚠️ A Note on the Dataset

Please note that the raw `dataset/` folder (containing all 639+ original PDFs) has **not** been uploaded to this repository. This is intentional to keep the repository lightweight and private.

The app does not need the raw data to run. All necessary information (the 1,278+ vector embeddings) has been pre-computed and is stored in the **`profiles/multivector_database.pkl`** file, which *is* included in this repository.

If you wish to re-build the database, you must:
1.  Place your own `dataset/` folder (with author subfolders) in the root directory.
2.  Run `python -m src.embed` (Note: This will take 20-30 minutes).

---

## 📁 File Structure Explained

| File | Purpose |
| :--- | :--- |
| **`app_streamlit.py`** | The main Streamlit frontend. It runs the UI, handles file uploads, and calls the similarity engine. |
| **`src/parse_pdf.py`** | A utility module that contains the core `extract_multivector_text` function to parse PDFs into "Best Text" and "Full Text". |
| **`src/embed.py`** | The **one-time setup script** used to build the database. It parses all PDFs in the `dataset/` folder and creates the final `.pkl` file. |
| **`src/similarity.py`** | The **recommender engine**. It loads the database and runs the "True Fallback" dual-search logic. |
| **`profiles/multivector_database.pkl`** | The **"brain"** of the app. This single file contains the pre-computed `best_text` and `full_text` embeddings for all 639 papers. |
| **`requirements.txt`** | The list of Python packages needed for Streamlit to install. |
| **`.gitignore`** | Tells Git to ignore the `dataset/`, `.venv/`, and other temporary files. |
