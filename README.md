# IT Infrastructure Q&A Assistant

This project is an interactive Question-Answering (QA) assistant for IT Infrastructure topics. It uses a Retrieval-Augmented Generation (RAG) pipeline powered by LangChain, Ollama LLM, and a custom retriever that combines local PDFs and web URLs.

## Features

- Answers IT Infrastructure questions using both local documents and online resources.
- Uses the Ollama LLM (`llama3.2` by default).
- Retrieves relevant information from PDFs and web pages.
- Interactive command-line interface.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running locally with the `llama3.2` model pulled
- Required Python packages (see below)
- Local PDFs in the `./data` directory

## Installation

1. **Clone the repository**  
   ```sh
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Install dependencies**  
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up Ollama**  
   - Download and install Ollama from [https://ollama.com/](https://ollama.com/)
   - Pull the required model:  
     ```sh
     ollama pull llama3.2
     ```

4. **Prepare your data**  
   - Place your PDF files inside the `./data` directory.

## Usage

Run the application:

```sh
python app.py
```

You’ll see a prompt to ask your IT Infrastructure questions. Type your question and press Enter. Type `q` to quit.

## Configuration

- **PDF Directory:**  
  Change the `pdf_directory` variable in `app.py` to point to your PDF folder.

- **Web URLs:**  
  Edit the `urls` list in `app.py` to add or remove online resources.

- **Ollama Model:**  
  Change the `model` variable if you want to use a different LLM.

## File Structure

```
app.py              # Main application file
vector.py           # Custom retriever logic
/data/              # Folder containing PDF documents
chroma_db/          # Vector database (auto-created)
requirements.txt    # Python dependencies
README.md           # Project documentation
```

## Notes

- Make sure Ollama is running before starting the app.
- The retriever combines information from both local PDFs and specified URLs.
- If you encounter issues, check that all dependencies are installed and the model is available in Ollama.

