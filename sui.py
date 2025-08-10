# app.py
import os
import time
from threading import Thread
from queue import Queue
from langchain_ollama.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from vector import pdf_url_rag
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.layout import Layout

# UI Setup
console = Console()
message_queue = Queue()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
PDF_DIR = "./data"
URLS = [
      "https://www.dell.com/en-us/lp/dt/end-user-computing",
"https://www.nutanix.com/solutions/end-user-computing",
"https://eucscore.com/docs/tools.html"
]

def loading_animation(message):
    """Show animated loading spinner"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=message, total=None)
        while not message_queue.empty():
            time.sleep(0.1)

def initialize_llm():
    """Initialize Ollama with error handling"""
    try:
        message_queue.put("Connecting to Ollama...")
        llm = Ollama(
            model=OLLAMA_MODEL,
            temperature=0.3,
            top_p=0.9,
            timeout=120
        )
        message_queue.put("✅ Model loaded successfully")
        return llm
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print("\n[bold]Make sure Ollama is running:[/bold] [blue]ollama serve[/blue]")
        exit(1)

def initialize_retriever():
    """Initialize vector store with visual feedback"""
    try:
        message_queue.put("Loading knowledge base...")
        retriever = pdf_url_rag(
            pdf_directory=PDF_DIR,
            urls=URLS,
            db_path="chroma_db",
            collection_name="tech_docs",
            model_name="nomic-embed-text"  # Faster alternative
        )
        message_queue.put("✅ Knowledge base ready")
        return retriever
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        exit(1)

def main():
    """Main application loop"""
    # Clear screen and show header
    console.clear()
    console.print(Panel.fit("📚 [bold blue]RAG Assistant[/]", border_style="blue"))

    # Initialize components with loading indicators
    Thread(target=loading_animation, args=("Starting up...",)).start()
    llm = initialize_llm()
    retriever = initialize_retriever()
    message_queue.put("done")  # Stop loading animation

    # Prepare the chain
    template = """You are a technical assistant. Answer based ONLY on:
    [Context]
    {documents}

    [Question]
    {question}

    If unsure, say "I don't know"."""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    # Main interaction loop
    while True:
        try:
            question = console.input("\n[bold cyan]Ask a question (q to quit):[/] ").strip()
            if question.lower() in ('q', 'quit', 'exit'):
                break

            if not question:
                console.print("[yellow]Please enter a question[/yellow]")
                continue

            # Process question with visual feedback
            Thread(target=loading_animation, args=("Thinking...",)).start()
            
            docs = retriever.invoke(question)
            answer = chain.invoke({"documents": docs, "question": question})
            
            message_queue.put("done")
            console.print(Panel(
                Markdown(answer),
                title="[green]Answer[/]",
                border_style="green"
            ))

        except KeyboardInterrupt:
            break
        except Exception as e:
            message_queue.put("done")
            console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    main()
    
    
    
    
    
    # app.py
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