# vector.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
from langchain_community.document_loaders import WebBaseLoader
import os
import time

ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

def pdf_url_rag(
    pdf_directory=None,
    urls=None,
    db_path="chroma_db",
    collection_name="documents",
    model_name: str = "mxbai-embed-large",
    chunk_size=1000, 
    chunk_overlap=150
):

    start_time = time.time()
    add_documents = not os.path.exists(db_path)
    embeddings = OllamaEmbeddings(model=model_name, base_url=ollama_url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if add_documents:
        print(f"DB is not created, creating db @ {db_path}")
        documents = []
        
        # Process PDFs if directory is provided
        if pdf_directory:
            for root, dirs, files in os.walk(pdf_directory):
                for filename in files:
                    if filename.endswith(".pdf"):
                        filepath = os.path.join(root, filename)
                        try:
                            loader = UnstructuredPDFLoader(filepath)
                            documents.extend(loader.load())
                            print(f"Loaded PDF: {filename}")
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
        
        # Process URLs if provided
        if urls:
            try:
                loader = WebBaseLoader(urls)
                web_docs = loader.load()
                documents.extend(web_docs)
                print(f"Loaded {len(urls)} URLs")
            except Exception as e:
                print(f"Error loading URLs: {str(e)}")

        if not documents:
            raise ValueError("No documents were loaded from either PDFs or URLs")
            
        texts = text_splitter.split_documents(documents)
        print(f"Adding {len(texts)} document chunks to the vector store...")
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=db_path,
            collection_name=collection_name
        )
        end_time = time.time()
        print(f"Processed documents in {end_time - start_time:.2f} seconds.")
        existing_ids = set(vector_store.get(include=[])['ids'])
        print(f"Total document chunks in store: {len(existing_ids)}")
    else:
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=embeddings
        )
        existing_ids = set(vector_store.get(include=[])['ids'])
        print(f"Total document chunks in store: {len(existing_ids)}")

    return vector_store.as_retriever(search_kwargs={"k": 2})