# from langchain_text_splitters import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain.docstore.document import Document
# import pandas as pd
# import os
# import glob

# # Define a function to perform vectorization for multiple CSV files
# def vectorize_documents():
#     embeddings = HuggingFaceEmbeddings()

#     # Directory containing multiple CSV files
#     csv_directory = "Data"  # Replace with your folder name
#     csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))  # Find all CSV files in the folder

#     documents = []

#     # Load and concatenate all CSV files
#     for file_path in csv_files:
#         df = pd.read_csv(file_path)
#         for _, row in df.iterrows():
#             # Combine all columns in the row into a single string
#             row_content = " ".join(row.astype(str))
#             documents.append(Document(page_content=row_content))

#     # Splitting the text and creating chunks of these documents
#     text_splitter = CharacterTextSplitter(
#         chunk_size=2000,
#         chunk_overlap=500
#     )

#     text_chunks = text_splitter.split_documents(documents)

#     # Process text chunks in batches
#     batch_size = 5000  # Chroma's batch size limit is 5461, set a slightly smaller size for safety
#     for i in range(0, len(text_chunks), batch_size):
#         batch = text_chunks[i:i + batch_size]

#         # Store the batch in Chroma vector DB
#         vectordb = Chroma.from_documents(
#             documents=batch,
#             embedding=embeddings,
#             persist_directory="vector_db_dir"
#         )

#     print("Documents Vectorized and saved in VectorDB")

# # Expose embeddings if needed
# embeddings = HuggingFaceEmbeddings()



# # Main guard to prevent execution on import
# if __name__ == "__main__":
#     vectorize_documents()



from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import pandas as pd
import os
import glob
from PyPDF2 import PdfReader  # Ensure PyPDF2 is installed

# Define a function to process CSV files
def process_csv_files(csv_files):
    documents = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            row_content = " ".join(row.astype(str))
            documents.append(Document(page_content=row_content))
    return documents

# Define a function to process PDF files
def process_pdf_files(pdf_files):
    documents = []
    for file_path in pdf_files:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:  # Only add non-empty text
                documents.append(Document(page_content=text))
    return documents

# Define a function to perform vectorization for CSV and PDF files
def vectorize_documents():
    embeddings = HuggingFaceEmbeddings()

    # Directory containing files
    data_directory = "Data"  # Replace with your folder name
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
    pdf_files = glob.glob(os.path.join(data_directory, "*.pdf"))

    # Process CSV and PDF files
    documents = process_csv_files(csv_files) + process_pdf_files(pdf_files)

    # Splitting the text and creating chunks of these documents
    text_splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500
    )

    text_chunks = text_splitter.split_documents(documents)

    # Process text chunks in batches
    batch_size = 5000  # Chroma's batch size limit is 5461, set a slightly smaller size for safety
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]

        # Store the batch in Chroma vector DB
        vectordb = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory="vector_db_dir"
        )

    print("Documents Vectorized and saved in VectorDB")

# Expose embeddings if needed
embeddings = HuggingFaceEmbeddings()

# Main guard to prevent execution on import
if __name__ == "__main__":
    vectorize_documents()
