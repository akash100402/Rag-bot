from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from vector import pdf_url_rag
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
import re

# Initialize Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Set user agent to prevent warnings
os.environ['USER_AGENT'] = 'ITInfraAssistant/1.0'

# Your existing RAG setup
ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
model = OllamaLLM(model="llama3.2", base_url=ollama_url)

# Modified template to explicitly prevent source references
template = """You are an expert IT infrastructure assistant. Answer the question using only the key information from the provided documents.
DO NOT mention document sources, metadata, or IDs in your response.
DO NOT include any references like (Source:...) or [doc1].
If you don't know the answer, say "I don't know".

Question: {question}

Relevant information: {documents}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Initialize retriever
retriever = pdf_url_rag(
    pdf_directory="./data",
    urls=[
        "https://www.dell.com/en-us/lp/dt/end-user-computing",
        "https://www.nutanix.com/solutions/end-user-computing",
        "https://eucscore.com/docs/tools.html"
    ],
    db_path="chroma_db",
    collection_name="mixed_docs"
)

def clean_response(text):
    """Final cleanup of response text"""
    # Remove any remaining document references
    text = re.sub(r'Document\(.*?\)', '', text)
    text = re.sub(r'id=.*?,', '', text)
    text = re.sub(r"metadata={.*?}", '', text)
    text = re.sub(r"page_content=.*?'", '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        docs = retriever.invoke(question)
        # Convert documents to clean text without metadata
        clean_docs = [doc.page_content for doc in docs]
        result = chain.invoke({"documents": clean_docs, "question": question})
        
        # Final cleanup
        clean_answer = clean_response(str(result))
        return jsonify({"answer": clean_answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve static files
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)