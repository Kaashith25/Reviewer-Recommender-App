import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import argparse

def load_database(path="profiles/multivector_database.pkl"):
    """Loads the pickled multivector database."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def find_similar_authors(query_best_vector, query_full_vector, st_model, database, top_k=5):
    """
    Performs the "True Fallback" dual-search.
    """
    
    # Prepare database matrices
    authors = []
    papers = []
    best_text_embeddings = []
    full_text_embeddings = []
    
    embedding_dim = st_model.get_sentence_embedding_dimension()
    
    for record in database:
        authors.append(record['author'])
        papers.append(record['paper'])
        
        # Add embedding or a vector of zeros if it's None
        if record['best_text_embedding'] is not None:
            best_text_embeddings.append(record['best_text_embedding'])
        else:
            best_text_embeddings.append(np.zeros(embedding_dim))
            
        if record['full_text_embedding'] is not None:
            full_text_embeddings.append(record['full_text_embedding'])
        else:
            full_text_embeddings.append(np.zeros(embedding_dim))

    # Convert to numpy matrices for fast computation
    db_best_text_matrix = np.array(best_text_embeddings)
    db_full_text_matrix = np.array(full_text_embeddings)
    
    # --- 1. Primary Search (Best vs. Best) ---
    score_A = cosine_similarity([query_best_vector], db_best_text_matrix).flatten()
    
    # --- 2. Fallback Search (Full vs. Full) ---
    score_B = cosine_similarity([query_full_vector], db_full_text_matrix).flatten()
    
    # --- 3. Get Final Score ---
    # For each paper, take the MAX score from either search
    final_scores = np.maximum(score_A, score_B)
    
    # --- 4. Aggregate by Author ---
    df = pd.DataFrame({
        'author': authors,
        'paper': papers,
        'score': final_scores,
    })
    
    author_scores = df.groupby('author')['score'].agg(['mean', 'count']).reset_index()
    author_max_scores = df.groupby('author')['score'].max().reset_index().rename(columns={'score': 'max'})
    author_scores = pd.merge(author_scores, author_max_scores, on='author')
    
    # Default sort by 'max'
    author_scores = author_scores.sort_values(by='max', ascending=False)

    return author_scores.head(top_k)


# Main execution block for command-line testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5, help="Number of authors to recommend")
    args = parser.parse_args()

    print("Loading model and MULTIVECTOR database...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    db_path = Path("profiles/multivector_database.pkl")
    if not db_path.exists():
        print(f"Error: {db_path} not found. Run 'python -m src.embed' first.")
        exit()
    database = load_database(str(db_path))

    # In test mode, we just use the same text for both queries
    print("Embedding query text...")
    query_best_vector = st_model.encode(args.query_text)
    query_full_vector = query_best_vector # Good enough for a simple test

    results = find_similar_authors(query_best_vector, query_full_vector, st_model, database, top_k=args.top_k)
    
    print(f"\nTop {args.top_k} Recommended Reviewers (Multivector Score):\n")
    print(results.to_string(index=False, float_format="%.4f"))