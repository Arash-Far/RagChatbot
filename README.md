# RAG Chatbot

A conversational AI chatbot built with LangGraph and FastAPI that uses Retrieval-Augmented Generation (RAG) to provide accurate responses based on a knowledge base. LangGraph provides key advantages out of the box including a persistence layer for maintaining conversation state and native streaming support for real-time responses.

## Features

- 🤖 Intelligent chatbot with RAG capabilities
- 🔍 FAISS vector store for efficient similarity search
- 🌐 FastAPI backend with streaming responses
- 💻 Streamlit frontend for easy interaction
- 🔄 Fallback to general responses when no relevant context is found

## Architecture

The application consists of:

- FastAPI backend server handling chat requests and RAG processing
- Streamlit frontend for user interface
- FAISS vector store for document retrieval
- LangGraph for orchestrating the chat flow
- OpenAI embeddings and GPT-4 for natural language processing

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:

bash
git clone https://github.com/yourusername/RagChatbot.git
cd RagChatbot

2. Install dependencies:

bash
pip install -r requirements.txt

3. Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the FastAPI backend server:

```bash
uvicorn server:app --reload --port 8000
```

2. In a new terminal, start the Streamlit frontend:

```bash
streamlit run client.py
```

3. Open your browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Configuration

- Modify `qa_data.json` to customize the knowledge base
- Adjust similarity search parameters in `server.py`
- Configure CORS settings in `server.py` if needed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
