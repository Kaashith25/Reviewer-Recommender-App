import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import argparse

# Import our new multivector extraction function
try:
    from src.parse_pdf import extract_multivector_text
except ImportError:
    print("Error: Could not find 'extract_multivector_text' in 'src/parse_pdf.py'")
    exit()

def build_multivector_database(dataset_dir="dataset", out_file="profiles/multivector_database.pkl"):
    """
    Creates the new multivector database.
    This will take a long time to run.
    """
    dataset_dir = Path(dataset_dir)
    pdf_files = sorted(list(dataset_dir.glob('*/*.pdf')))
    
    if not pdf_files:
        print(f"Error: No .pdf files found in {dataset_dir} subfolders.")
        return

    print("Loading Sentence Transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # This will be a list of dictionaries
    database_records = []
    
    print(f"Parsing and embedding {len(pdf_files)} PDFs...")
    print("This will take 20-30 minutes.")

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        author_name = pdf_path.parent.name
        paper_name = f"{author_name}__{pdf_path.stem}.pdf"
        
        # 1. Extract both text types
        texts = extract_multivector_text(str(pdf_path))

        # 2. Embed 'best_text'
        best_text_embedding = None
        if texts['best_text']:
            best_text_embedding = model.encode(texts['best_text'], convert_to_numpy=True)
            
        # 3. Embed 'full_text'
        full_text_embedding = None
        if texts['full_text']:
            full_text_embedding = model.encode(texts['full_text'], convert_to_numpy=True)
        
        # 4. Add to our database
        database_records.append({
            'author': author_name,
            'paper': paper_name,
            'best_text_embedding': best_text_embedding,
            'full_text_embedding': full_text_embedding
        })

    print(f"\nCreated {len(database_records)} records.")
    
    # 5. Save the final database
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(database_records, f)
    
    print(f"\n✅ New 'Multivector Database' saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--out', default='profiles/multivector_database.pkl')
    args = parser.parse_args()
    build_multivector_database(args.dataset, args.out)