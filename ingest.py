import os
from docx import Document
from pypdf import PdfReader

DOCS_PATH = "./docs"


def read_docx(path):
    doc = Document(path)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])


def read_pdf(path):
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def read_file(path):
    if path.endswith(".docx"):
        return read_docx(path)
    elif path.endswith(".pdf"):
        return read_pdf(path)
    else:
        print("Skipping unsupported file:", path)
        return None


def load_all_documents(folder_path):
    documents = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            text = read_file(file_path)

            if text and text.strip():
                documents[file_path] = text

    return documents


if __name__ == "__main__":
    docs = load_all_documents(DOCS_PATH)

    print("Documents loaded:", len(docs))

    for name in docs:
        print("-", name)