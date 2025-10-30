import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
import argparse
import re

def clean_full_text(text):
    """
    Finds the 'References' or 'Bibliography' section and cuts it off.
    """
    text_lower = text.lower()
    
    # Find the last occurrence of "references" or "bibliography"
    # We add \n to ensure it's a section heading
    ref_pos = text_lower.rfind('\nreferences')
    bib_pos = text_lower.rfind('\nbibliography')
    
    # Find the position of the last marker
    split_pos = max(ref_pos, bib_pos)
    
    if split_pos != -1:
        # We found a marker, so cut the text off there
        return text[:split_pos]
    else:
        # No marker found, return the whole text
        return text

def extract_multivector_text(path):
    """
    The main parsing function.
    Extracts two versions of text from a PDF:
    1. 'best_text': The Abstract + Introduction.
    2. 'full_text': The cleaned full text (no references).
    """
    full_text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"Warning: Could not parse {path}. {e}")
        return {'best_text': None, 'full_text': None}

    if not full_text.strip():
        return {'best_text': None, 'full_text': None}

    # --- 1. Create the 'full_text' (Cleaned) ---
    # This is our fallback
    cleaned_full_text = clean_full_text(full_text)
    
    # --- 2. Create the 'best_text' (Abstract + Intro) ---
    best_text = ""
    full_text_lower = full_text.lower()
    
    # Try to find "Abstract"
    abstract_start_match = re.search(r'(?<!in\s)abstract', full_text_lower)
    
    if abstract_start_match:
        abstract_start = abstract_start_match.start()
        
        # Find where the abstract *ends* (usually before intro/keywords)
        abstract_end_match = re.search(r'introduction|keywords|\n1\.', full_text_lower[abstract_start:])
        
        if abstract_end_match:
            abstract_end = abstract_start + abstract_end_match.start()
            best_text += full_text[abstract_start:abstract_end]
        else:
            # Couldn't find the end, just take 500 words
            best_text += " ".join(full_text[abstract_start:].split()[:500])

    # Try to find "Introduction" (whether we found abstract or not)
    intro_start_match = re.search(r'introduction', full_text_lower)
    
    if intro_start_match:
        intro_start = intro_start_match.start()
        
        # Find where the intro *ends* (usually before methods/related work/section 2)
        intro_end_match = re.search(r'methods|related work|background|\n2\.', full_text_lower[intro_start:])
        
        if intro_end_match:
            intro_end = intro_start + intro_end_match.start()
            best_text += "\n" + full_text[intro_start:intro_end]
        else:
            # Couldn't find the end, just take 1000 words
            best_text += "\n" + " ".join(full_text[intro_start:].split()[:1000])

    # --- 3. Final Check ---
    # If best_text is still tiny (less than 100 words), it failed. Set to None.
    if len(best_text.split()) < 100:
        best_text = None
        
    return {
        'best_text': best_text, 
        'full_text': cleaned_full_text
    }


if __name__ == '__main__':
    """
    This is just for testing. The main embedding script is now 'embed.py'
    """
    p = argparse.ArgumentParser(description="Test the multivector text extraction on a single PDF.")
    p.add_argument('--pdf', required=True, help="Path to a single PDF file to test.")
    args = p.parse_args()
    
    texts = extract_multivector_text(args.pdf)
    
    print("--- BEST TEXT (Abstract + Intro) ---")
    if texts['best_text']:
        print(texts['best_text'][:1000] + "...")
    else:
        print("COULD NOT FIND BEST TEXT.")
        
    print("\n\n--- FULL TEXT (Cleaned) ---")
    if texts['full_text']:
        print(texts['full_text'][:1000] + "...")
    else:
        print("COULD NOT FIND FULL TEXT.")