<div align="center">
  <h1>📚 RAG Pipeline: Complete Guide from Ingestion to Retrieval</h1>
  <p>
    <b>A comprehensive Retrieval-Augmented Generation (RAG) system built with Python, LangChain, and ChromaDB</b>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/LangChain-Framework-green.svg" alt="LangChain">
    <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg" alt="ChromaDB">
    <img src="https://img.shields.io/badge/SentenceTransformers-Embeddings-yellow.svg" alt="Embeddings">
  </p>
</div>

<hr>

## 📖 Table of Contents
- [What is RAG?](#-what-is-rag)
- [Project Architecture](#️-project-architecture)
- [Key Modules & Why We Use Them](#️-key-modules--why-we-use-them)
- [Detailed Code Walkthrough](#-detailed-code-walkthrough)
- [Folder Structure](#-folder-structure)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [How to Use](#-how-to-use)

<hr>

## 🤖 What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI technique that combines two powerful capabilities:
1. **Retrieval**: Finding relevant information from a knowledge base
2. **Generation**: Using that information to generate accurate responses

### Why RAG?
Traditional Large Language Models (LLMs) have two major limitations:
- ❌ **Knowledge Cutoff**: They only know information up to their training date
- ❌ **Hallucinations**: They sometimes generate incorrect information confidently

RAG solves these problems by:
- ✅ Fetching **real-time, domain-specific data** from your own documents
- ✅ Grounding responses in **actual source material**, reducing hallucinations
- ✅ Making AI systems **verifiable** - you can trace answers back to source documents

### How Does RAG Work?

Think of RAG like a smart research assistant:
1. **You have a library** (your documents/PDFs)
2. **You ask a question** ("What are the benefits of exception handling?")
3. **The assistant searches** the library for relevant sections
4. **The assistant reads** those sections and answers your question based on what it found

<div align="center">
  <table>
    <tr>
      <td align="center"><b>📥 Data Ingestion</b></td>
      <td align="center">→</td>
      <td align="center"><b>✂️ Chunking</b></td>
      <td align="center">→</td>
      <td align="center"><b>🧮 Embedding</b></td>
      <td align="center">→</td>
      <td align="center"><b>💾 Vector Store</b></td>
      <td align="center">→</td>
      <td align="center"><b>🔍 Retrieval</b></td>
    </tr>
  </table>
</div>

<hr>

## 🏗️ Project Architecture

This project implements the complete RAG pipeline step-by-step:

### 1. **Data Ingestion** 📥
- **What**: Loading raw documents (PDFs, text files) into Python
- **Why**: We need to bring external knowledge into our system
- **How**: Using LangChain's document loaders (`TextLoader`, `PyMuPDFLoader`)

### 2. **Text Chunking** ✂️
- **What**: Splitting large documents into smaller pieces (chunks)
- **Why**: 
  - LLMs have input size limits
  - Smaller chunks make retrieval more precise
  - We want to find the *specific paragraph* that answers the question, not the entire 100-page document
- **How**: Using `RecursiveCharacterTextSplitter` with:
  - **Chunk Size**: 1000 characters per chunk
  - **Chunk Overlap**: 200 characters overlap between chunks (preserves context across boundaries)

### 3. **Embedding Generation** 🧮
- **What**: Converting text into numerical vectors (arrays of numbers)
- **Why**: Computers can't directly compare "meaning" of text, but they can compare vectors using math
- **Example**: 
  - "Python programming" → `[0.23, -0.45, 0.67, ...]` (384 numbers)
  - "Coding in Python" → `[0.25, -0.43, 0.69, ...]` (very similar numbers!)
- **How**: Using `SentenceTransformer` model (`all-MiniLM-L6-v2`)

### 4. **Vector Storage** 💾
- **What**: Saving embeddings in a specialized database
- **Why**: We need fast similarity search across thousands/millions of vectors
- **How**: Using ChromaDB - a persistent vector database

### 5. **Retrieval** 🔍
- **What**: Finding the most relevant chunks for a user's query
- **Why**: We only want to show the LLM the top 5 most relevant passages, not all 10,000 chunks
- **How**: Convert query to embedding → Find nearest neighbors using cosine similarity

<hr>

## 🛠️ Key Modules & Why We Use Them

Here's a detailed explanation of every library and module used in this project:

<table>
  <thead>
    <tr>
      <th>Module / Library</th>
      <th>Purpose</th>
      <th>Detailed Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><code>langchain_core.documents</code></b></td>
      <td>Document Structure</td>
      <td>Provides the <code>Document</code> class which is the foundation for storing text data along with metadata (source file, page number, author, etc.). Every piece of text in our system is wrapped in this format.</td>
    </tr>
    <tr>
      <td><b><code>langchain_community.document_loaders</code></b></td>
      <td>File Loading</td>
      <td>
        Contains specialized loaders for different file types:<br>
        • <code>TextLoader</code>: Reads .txt files with proper encoding<br>
        • <code>DirectoryLoader</code>: Batch loads all files matching a pattern<br>
        • <code>PyMuPDFLoader</code>: Extracts text from PDFs while preserving page numbers and metadata
      </td>
    </tr>
    <tr>
      <td><b><code>langchain_text_splitters</code></b></td>
      <td>Text Chunking</td>
      <td>
        <code>RecursiveCharacterTextSplitter</code> intelligently splits text by:<br>
        1. First trying to split on paragraphs (<code>\n\n</code>)<br>
        2. Then sentences (<code>\n</code>)<br>
        3. Then words (<code> </code>)<br>
        4. Finally characters if needed<br>
        This preserves natural language structure better than random character splits.
      </td>
    </tr>
    <tr>
      <td><b><code>sentence_transformers</code></b></td>
      <td>Embedding Model</td>
      <td>
        Uses pre-trained neural networks to convert text into dense vectors:<br>
        • Model: <code>all-MiniLM-L6-v2</code> (384 dimensions)<br>
        • Fast and efficient (6 layers)<br>
        • Works well for semantic similarity tasks<br>
        • Pre-trained on millions of sentence pairs from the internet
      </td>
    </tr>
    <tr>
      <td><b><code>chromadb</code></b></td>
      <td>Vector Database</td>
      <td>
        A specialized database for storing and querying embeddings:<br>
        • <b>Persistent storage</b>: Saves to disk, no need to re-embed each time<br>
        • <b>Fast similarity search</b>: Uses approximate nearest neighbor algorithms<br>
        • <b>Collections</b>: Organizes embeddings into groups (like database tables)<br>
        • Automatically handles distance calculations (cosine, euclidean, etc.)
      </td>
    </tr>
    <tr>
      <td><b><code>uuid</code></b></td>
      <td>Unique Identifiers</td>
      <td>
        Generates universally unique IDs for each document chunk:<br>
        • Format: <code>doc_a3f5b89c_42</code><br>
        • Ensures no two chunks have the same ID, even across multiple runs<br>
        • Required by ChromaDB for tracking documents
      </td>
    </tr>
    <tr>
      <td><b><code>sklearn.metrics.pairwise</code></b></td>
      <td>Similarity Calculation</td>
      <td>
        <code>cosine_similarity</code> measures how similar two vectors are:<br>
        • Returns a score from -1 to 1<br>
        • 1 = identical meaning<br>
        • 0 = unrelated<br>
        • -1 = opposite meaning
      </td>
    </tr>
  </tbody>
</table>

<hr>

## 💻 Detailed Code Walkthrough

Let's break down each major component of the code in `notebook/document.ipynb`:

### 📌 Part 1: Data Ingestion

#### Creating Sample Documents
```python
doc = Document(
    page_content="this is the main text content",
    metadata={"source": "example.txt", "author": "predator"}
)
```
**What's happening?**
- We create a `Document` object (LangChain's standard format)
- `page_content`: The actual text we want to store
- `metadata`: Extra information (where it came from, who wrote it, etc.)

#### Loading Text Files
```python
loader = TextLoader("../data/txt_files/python_intro.txt", encoding="utf-8")
documents = loader.load()
```
**What's happening?**
- `TextLoader` reads the file
- `encoding="utf-8"` ensures special characters (emojis, non-English text) are handled correctly
- Returns a list of `Document` objects (one per file)

#### Loading PDFs
```python
for path in glob.glob("../data/pdfs/*.pdf"):
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = path  # 🔑 CRITICAL!
    pdf_documents.extend(docs)
```
**What's happening?**
- `glob.glob` finds all PDF files in the folder
- `PyMuPDFLoader` extracts text from each page
- **Important**: We manually set `metadata["source"]` so we can trace answers back to the original PDF
- Result: 187 documents (one per page across all PDFs)

---

### 📌 Part 2: Text Splitting

```python
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs
```

**What's happening?**
- **`chunk_size=1000`**: Each chunk is ~1000 characters (roughly 2-3 paragraphs)
- **`chunk_overlap=200`**: Last 200 characters of Chunk A are repeated at the start of Chunk B
  - **Why?** If a sentence is cut between chunks, it still appears complete in one chunk
- **`separators`**: Try to split at paragraph boundaries first, then newlines, then spaces
- **Result**: 187 documents → 481 chunks (smaller, more precise pieces)

---

### 📌 Part 3: Embedding Generation

#### The `EmbeddingManager` Class
```python
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
```

**What's happening?**
- **`__init__`**: Downloads and loads the embedding model (first run only, then cached)
- **`generate_embeddings`**: 
  - Takes a list of text strings
  - Returns a NumPy array of shape `(num_texts, 384)`
  - Each text becomes a 384-dimensional vector

**Example**:
```python
texts = ["Python is great", "I love programming"]
embeddings = embedding_manager.generate_embeddings(texts)
# Shape: (2, 384)
# embeddings[0] = [0.12, -0.45, 0.67, ...]  ← "Python is great"
# embeddings[1] = [0.15, -0.42, 0.71, ...]  ← "I love programming"
```

---

### 📌 Part 4: Vector Store

#### The `VectorStore` Class
```python
class VectorStore:
    def __init__(self, collection_name="pdf_documents", 
                 persist_directory="../data/vector_store"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
    def add_documents(self, documents, embeddings):
        ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        metadatas = [dict(doc.metadata) for doc in documents]
        documents_text = [doc.page_content for doc in documents]
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text
        )
```

**What's happening?**
- **`PersistentClient`**: Creates/opens a database at `../data/vector_store`
  - Data persists between runs (no need to re-embed every time!)
- **`get_or_create_collection`**: Like creating a table in SQL
- **`add_documents`**:
  - Generates unique IDs for each chunk
  - Stores 4 things: ID, embedding vector, metadata, original text
  - ChromaDB automatically indexes the embeddings for fast search

---

### 📌 Part 5: Retrieval

#### The `RAGRetriever` Class
```python
class RAGRetriever:
    def retrieve(self, query, top_k=5, score_threshold=0.0):
        # Step 1: Convert query to embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Step 2: Search vector store
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Step 3: Process results
        retrieved_docs = []
        for i, (doc_id, document, metadata, distance) in enumerate(...):
            similarity_score = 1 - distance  # Convert distance to similarity
            if similarity_score >= score_threshold:
                retrieved_docs.append({
                    'content': document,
                    'metadata': metadata,
                    'similarity_score': similarity_score
                })
        
        return retrieved_docs
```

**What's happening?**
1. **Query → Embedding**: Convert your question to a vector (same 384 dimensions)
2. **Vector Search**: ChromaDB finds the 5 closest vectors using cosine distance
3. **Score Conversion**: Distance is converted to similarity (higher = better match)
4. **Filtering**: Only return results above the threshold

**Example Query Flow**:
```
User Query: "What are the benefits of exception handling?"
           ↓
Embedding:  [0.34, -0.12, 0.89, ...] (384 numbers)
           ↓
ChromaDB Search: Find nearest 5 vectors
           ↓
Results:   
  1. "Exception handling provides..." (similarity: 0.92)
  2. "Benefits of try-catch blocks..." (similarity: 0.87)
  3. ...
```

---

## 📂 Folder Structure

```
RAG/
├── 📁 data/
│   ├── 📁 pdfs/                    # Your source PDF documents
│   │   ├── JavaInterviewQuestions.pdf
│   │   ├── EngineeringPhysics.pdf
│   │   └── ExceptionHandling.pdf
│   ├── 📁 txt_files/               # Your source text files
│   │   ├── python_intro.txt
│   │   └── ml_intro.txt
│   └── 📁 vector_store/            # ChromaDB persistent storage
│       ├── chroma.sqlite3          # Database file
│       └── ...                     # Index files
│
├── 📁 notebook/
│   └── document.ipynb              # Main RAG pipeline (Jupyter Notebook)
│
├── 📁 .venv/                       # Virtual environment (isolated Python packages)
│
├── .gitignore                      # Git ignore file
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
└── README.md                       # This file!
```

---

## 🔧 Tech Stack

| Technology | Purpose |
|:-----------|:--------|
| **Python 3.10+** | Programming language |
| **LangChain** | RAG framework & document processing |
| **ChromaDB** | Vector database for embeddings |
| **SentenceTransformers** | Neural embedding model |
| **PyMuPDF** | PDF text extraction |
| **Jupyter Notebook** | Interactive development environment |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher installed on your system
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd RAG
```

### Step 2: Create a Virtual Environment

A **virtual environment** is an isolated Python environment for your project. It prevents dependency conflicts between different projects.

#### On Windows:
```bash
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# You'll see (.venv) appear in your terminal prompt
```

#### On Mac/Linux:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# You'll see (.venv) appear in your terminal prompt
```

**What just happened?**
- `python -m venv .venv`: Creates a new folder `.venv` containing a copy of Python and pip
- `activate`: Switches your terminal to use THIS copy of Python
- Now when you run `pip install`, packages go into `.venv`, not your system Python

**To deactivate later:**
```bash
deactivate
```

### Step 3: Install Dependencies
```bash
# Make sure your virtual environment is activated!
pip install -r requirements.txt
```

This installs:
- `langchain`
- `langchain-community`
- `chromadb`
- `sentence-transformers`
- `pymupdf`
- And all their dependencies

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```
- This opens a web browser
- Navigate to `notebook/document.ipynb`
- Run cells sequentially (Shift+Enter)

---

## 🎯 How to Use

### Running the Complete Pipeline

1. **Activate your virtual environment**:
   ```bash
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Mac/Linux
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook notebook/document.ipynb
   ```

3. **Run all cells in order** (Cell → Run All)

4. **Query your documents**:
   ```python
   # At the end of the notebook
   results = rag_retriever.retrieve("Your question here")
   
   # Print clean results
   for doc in results:
       print(f"Content: {doc['content']}")
       print(f"Source: {doc['metadata']['source']}")
       print(f"Similarity: {doc['similarity_score']:.2f}")
       print("-" * 50)
   ```

### Adding Your Own Documents

1. **For PDFs**: Drop them in `data/pdfs/`
2. **For text files**: Drop them in `data/txt_files/`
3. **Re-run the notebook** to ingest and embed the new documents

### Understanding Chunk Overlap

If you notice your retrieval results include text from "before" your topic:
- This is **by design** due to `chunk_overlap=200`
- The overlap preserves context across chunk boundaries
- You can reduce it by changing `chunk_overlap` in the `split_documents` function
- Trade-off: Less overlap = faster, but might lose context

---

<div align="center">
  <h3>🎉 You now have a fully functional RAG system!</h3>
  <p><i>Built with ❤️ for learning and exploration</i></p>
</div>

<hr>

## 📝 License
This project is open source and available for educational purposes.

## 🤝 Contributing
Feel free to fork, modify, and submit pull requests!

## 📧 Contact
Questions? Feedback? Open an issue on GitHub!
