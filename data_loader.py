import os
import pandas as pd

def load_pubmed_20k_rct():
    data_dir = "pubmed-rct/PubMed_20k_RCT"
    
    splits = ["dev", "test", "train"]  # Use ["dev"] for quick test
    
    titles = []
    abstracts = []
    
    for split in splits:
        path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(path):
            print(f"Warning: {path} not found!")
            continue
        
        print(f"Loading {split}.txt...")
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        current_title = ""
        current_abstract = ""
        for line in lines:
            line = line.strip()
            if line.startswith("###"):
                if current_title:
                    titles.append(current_title.strip())
                    abstracts.append(current_abstract.strip())
                current_title = ""
                current_abstract = ""
            elif "\t" in line:
                role, text = line.split("\t", 1)
                text = text.strip()
                if role == "title":
                    current_title = text
                else:
                    current_abstract += text + " "
        
        if current_title:
            titles.append(current_title.strip())
            abstracts.append(current_abstract.strip())
    
    if not titles:
        raise ValueError("No titles parsed — check file paths.")
    
    df = pd.DataFrame({
        "title": titles,
        "abstract": abstracts
    })
    
    print(f"Total abstracts loaded: {len(df)}")
    
    # Filter for gut/microbiome
    keywords = ["microbiome", "gut", "intestinal", "microbiota", "probiotic", "prebiotic", "dysbiosis", "flora", "stool", "feces"]
    pattern = '|'.join(keywords)
    mask = df['abstract'].str.contains(pattern, case=False, na=False)
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        print("No matches — saving all for testing.")
        filtered_df = df
    
    filtered_df.to_csv("microbiome_abstracts.csv", index=False)
    print(f"Saved {len(filtered_df)} abstracts with titles!")
    print("\nSample:")
    print(filtered_df[['title', 'abstract']].head(2))
    
    return filtered_df

if __name__ == "__main__":
    load_pubmed_20k_rct()
