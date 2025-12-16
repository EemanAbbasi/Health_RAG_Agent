import os
import pandas as pd

def load_pubmed_20k_rct():
    data_dir = "pubmed-rct/PubMed_20k_RCT"
    
    splits = ["dev", "test", "train"]  # Start with ["dev"] for fast test
    
    all_texts = []
    
    for split in splits:
        path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(path):
            print(f"Warning: {path} not found!")
            continue
        
        print(f"Loading {split}.txt...")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by paper (###ID)
        papers = content.split("###")[1:]  # Skip first empty
        
        for paper in papers:
            lines = paper.strip().split("\n")
            if not lines:
                continue
            # First line after ### is usually ID, skip it
            text_lines = lines[1:]
            full_text = " ".join(line.strip() for line in text_lines if "\t" in line or line.strip())
            if full_text:
                all_texts.append(full_text)
    
    if not all_texts:
        raise ValueError("No text parsed — files may be empty or format different.")
    
    df = pd.DataFrame({
        "abstract": all_texts
    })
    
    print(f"Total abstracts loaded: {len(df)}")
    
    # Filter for gut/microbiome
    keywords = ["microbiome", "gut", "intestinal", "microbiota", "probiotic", "prebiotic", "dysbiosis", "flora", "stool", "feces"]
    pattern = '|'.join(keywords)
    mask = df['abstract'].str.contains(pattern, case=False, na=False)
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        print("No matches — saving first 500 abstracts for testing.")
        filtered_df = df.head(500)
    
    filtered_df.to_csv("microbiome_abstracts.csv", index=False)
    print(f"Saved {len(filtered_df)} abstracts!")
    print("\nSample:")
    print(filtered_df['abstract'].head(2))
    
    return filtered_df

if __name__ == "__main__":
    load_pubmed_20k_rct()