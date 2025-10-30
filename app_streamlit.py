import streamlit as st
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Import functions from our src files ---
try:
    # We need the multivector extraction function
    from src.parse_pdf import extract_multivector_text
except ImportError:
    st.error("Could not import 'extract_multivector_text'. Make sure it's in src/parse_pdf.py.")
    st.stop()

try:
    # Import our new multivector functions
    from src.similarity import (
        load_database, 
        find_similar_authors
    )
except ImportError:
    st.error("Could not import similarity functions. Make sure src/similarity.py is updated.")
    st.stop()


st.set_page_config(page_title='Reviewer Recommender', layout='wide')
st.title('Reviewer Recommender')

# --- NEW: DETAILED EXPLANATION OF THE SYSTEM ---
with st.expander("Click to see how this system works"):
    st.markdown("""
        This system is designed to be robust against poorly formatted PDFs. It runs **two separate, parallel searches** to ensure no relevant paper is missed, then takes the best score.

        1.  **Search #1: Abstract-to-Abstract (Primary)**
            * The system first extracts the **"Best Text"** (Abstract + Introduction) from your uploaded paper.
            * It compares this "Best Text" against the "Best Text" database of all 600+ papers.
            * This is an "apples-to-apples" comparison that provides highly accurate scores for well-formatted papers.

        2.  **Search #2: FullText-to-FullText (Fallback)**
            * The system also extracts the **"Full Text"** (cleaned) of your uploaded paper.
            * It compares this "Full Text" against the "Full Text" database.
            * This "oranges-to-oranges" search acts as a fallback, ensuring that if a relevant paper has no abstract, it is *still* found based on its full content.
        
        The final score for any paper is the **maximum (best) score** from either Search #1 or Search #2.
    """)

# --- Cache the models and data ---
MODEL_NAME = "all-MiniLM-L6-v2"
DATABASE_PATH = "profiles/multivector_database.pkl" 

@st.cache_resource
def get_sentence_model():
    """Loads and caches the SentenceTransformer model."""
    print("Loading SentenceTransformer model...")
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def get_multivector_database(path):
    """Loads and caches the multivector database."""
    db_path = Path(path)
    if not db_path.exists():
        st.error(f"Error: Multivector database not found at {path}.")
        st.error("Please run 'python -m src.embed' from your terminal. (This will take a long time)")
        return None
    print("Loading multivector database...")
    return load_database(str(db_path))

# --- Load all models ---
try:
    st_model = get_sentence_model()
    database = get_multivector_database(DATABASE_PATH)
except Exception as e:
    st.error(f"An error occurred during model loading: {e}")
    st.stop()

# --- Main App Logic ---
if database is not None:
    uploaded = st.file_uploader('Upload a PDF research paper', type=['pdf'])

    if uploaded:
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / uploaded.name
        
        try:
            tmp_path.write_bytes(uploaded.read())
            
            with st.spinner('Extracting both "Best Text" and "Full Text" from PDF...'):
                query_texts = extract_multivector_text(str(tmp_path))
            
            # --- Check if we got any valid text ---
            if not query_texts['best_text'] and not query_texts['full_text']:
                st.error("Failed to extract any text from this PDF.")
            else:
                
                # --- Show a preview of the best text ---
                if query_texts['best_text']:
                    st.subheader('Extracted Best Text (Abstract/Intro) Preview:')
                    st.info(query_texts['best_text'][:800] + '...')
                else:
                    st.warning("Could not find 'Best Text' (Abstract/Intro). Using Full-Text Fallback only.")

                # --- Sidebar for controls ---
                st.sidebar.subheader("Controls")
                top_k = st.sidebar.slider('How many reviewers?', 1, 20, 5)
                ranking_method = st.sidebar.radio(
                    "Ranking Method",
                    ('max', 'mean'),
                    index=0, # Default to 'max'
                    help="'max' ranks by the best single paper. 'mean' ranks by the author's average."
                )
                
                if st.button(f'Find Top {top_k} Reviewers'):
                    with st.spinner(f'Running dual-search and finding reviewers...'):
                        
                        # --- Embed BOTH query texts ---
                        query_best_vector = st_model.encode(query_texts['best_text'] or "")
                        query_full_vector = st_model.encode(query_texts['full_text'] or "")

                        res_df = find_similar_authors(
                            query_best_vector,
                            query_full_vector,
                            st_model,
                            database, 
                            top_k=top_k
                        )
                    
                    # Sort based on user's choice
                    res_df = res_df.sort_values(by=ranking_method, ascending=False)
                    
                    st.subheader('Recommended Reviewers:')

                    # --- NEW: EXPLANATION OF THE TABLE ---
                    with st.expander("How to read these results"):
                        st.markdown("""
                            * **author**: The name of the recommended reviewer.
                            * **max**: The **single best score** from one of that author's papers. A high `max` score means the author has at least one *highly relevant* "niche" paper.
                            * **mean**: The **average score** across *all* of that author's papers. A high `mean` score means the author's *entire research area* is consistently relevant to your topic.
                            * **count**: The total number of papers by this author in the database.
                        """)
                    
                    res_df_display = res_df.copy()
                    res_df_display['mean'] = res_df_display['mean'].map('{:,.4f}'.format)
    
                    if 'max' in res_df_display.columns:
                        res_df_display['max'] = res_df_display['max'].map('{:,.4f}'.format)
                    
                    # --- NEW: REORDERED COLUMNS ---
                    cols_to_display = ['author', 'max', 'mean', 'count']
                    st.dataframe(res_df_display[cols_to_display], use_container_width=True, hide_index=True)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
else:
    st.warning("Database could not be loaded. Please run 'python -m src.embed' and try again.")