import fitz, os
import argparse
# ------------------defaults------------------
data_dir = r"../../data/Papers"
knowledge_file = r"../../data/knowledge_try.txt"

#------------------utils------------------
def extract_text(data_dir: str, out_path: str) -> None:
    """
    Extracts and writes text from multiple PDF files in a specified directory to a text file. Works best for PDFs with minimal images.

    Args: 
        data_dir: Directory containing the PDF files.
        out_path: Path of text file to store the text.

    """

    all_text = ''
    for file in os.listdir(data_dir):
        if not file.endswith(".pdf"): print(f"Skipping {file}, not a PDF file"); continue

        filepath = os.path.join(data_dir, file)
        doc = fitz.open(filepath)
        for page in doc:
            all_text += page.get_text()
        doc.close()

    assert out_path.endswith(".txt"), "Can only save to a text file"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(all_text)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract text from PDF files in a directory and save to a text file.")
    parser.add_argument('--data_dir', type=str, default=data_dir, help='Directory containing PDF files')
    parser.add_argument('--out_path', type=str, default=knowledge_file, help='Path to save the extracted text file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    extract_text(args.data_dir, args.out_path)