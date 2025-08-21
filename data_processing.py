from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
import json

def process_text(text:str)-> List[str]:
    """Splits text into chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600, 
        chunk_overlap=120,
        length_function=len)
    
    chunks = text_splitter.split_text(text)
    return chunks


def process_pdfs(folder_path: Path) -> List[Dict]:
    """Extract text from all PDF files in a folder."""
    pdf_files = list(folder_path.glob("*.pdf"))
    all_chunks = []
    
    for i, pdf_file in tqdm(enumerate(pdf_files)):
        pdf_reader = PdfReader(pdf_file)
        text = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        pdf_reader.close()
        pdf_text =  "\n".join(text)
        chunks = process_text(pdf_text)
        for j, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": i,
                "filename": pdf_file.name,
                "chunk_id": j,
                "text": chunk
            })
    
    return all_chunks


def save_chunks_jsonl(chunks: List[Dict], output_file: Path):
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(chunks)} chunks to {output_file}")


def main():
    folder_path = Path("data")
    output_file = Path("data/chunks.jsonl")
    
    if not folder_path.exists():
        print(f"Folder {folder_path} does not exist.")
        return
    
    chunks = process_pdfs(folder_path)
    save_chunks_jsonl(chunks, output_file)

if __name__ == "__main__":
    main()