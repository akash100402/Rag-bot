from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import pdf_url_rag
import os

ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
model = OllamaLLM(model="llama3.2", base_url=ollama_url)

template = """
You are an expert in answering Information Technology Infra Structure questions and you will get answers from the provided documents.
If you don't know the answer, say "I don't know".

Here are some relevant documents: {documents}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Configuration
pdf_directory = "./data"  # Contains subfolders with PDFs
urls = [
   "https://www.dell.com/en-us/lp/dt/end-user-computing",
"https://www.nutanix.com/solutions/end-user-computing",
"https://eucscore.com/docs/tools.html"
]

retriever = pdf_url_rag(
    pdf_directory=pdf_directory,
    urls=urls,
    db_path="chroma_db",
    collection_name="mixed_docs"
)
print("Retriever ready.")

while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question.lower() == 'q':
        break
    
    docs = retriever.invoke(question)
    result = chain.invoke({"documents": docs, "question": question})
    print(f"AI: {result}")