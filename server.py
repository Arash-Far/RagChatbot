import json
import faiss
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from typing import Literal


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize embeddings and create vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load the predefined data from JSON
with open("qa_data.json", "r") as f:
    DATA = json.load(f)

# Convert Q&A into documents
documents = []
for qa in DATA["questions"]:
    doc = Document(
        page_content=qa["question"], 
        metadata={"answer": qa["answer"]}
    )
    documents.append(doc)

# Create FAISS index with the correct dimension
embedding_size = len(embeddings.embed_query("test"))
index = faiss.IndexFlatL2(embedding_size)

# Create vector store with explicit FAISS index
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Index documents to the vector store
_ = vector_store.add_documents(documents=documents)

# Initialize the model with streaming
model = ChatOpenAI(
    model="gpt-4",
    streaming=True,
    api_key=api_key
)

# Create prompt templates
thoughtful_ai_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Thoughtful AI support assistant. Use the following relevant information "
        "to answer the user's question about Thoughtful AI's products and services. Use only the answer from the context as your answer.\n\n"
        "Context: {context}"
    ),
    MessagesPlaceholder(variable_name="messages"),
])

general_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Please provide a general helpful response."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

class State(MessagesState):
    metadatas: list
    
    


def retrieve(state):

    messages = state["messages"]
    
    # Get the user's question from the last message
    if isinstance(messages[-1], HumanMessage):
        query = messages[-1].content
        
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 1})       
        docs = retriever.invoke(query)
        metadatas = [doc.metadata["answer"] for doc in docs]
        return {"metadatas": metadatas}
        

def handle_response(state) -> Literal["rag_response", "general_response"]:
    metadatas = state["metadatas"]
    messages = state["messages"]
    if len(metadatas) > 0:     
        return "rag_response"
    else:
        return "general_response"

    
def rag_response(state):
    metadatas = state["metadatas"]
    response_text = "\n\n".join(metadatas)
    response = AIMessage(content=response_text)
    return {"messages": [response]}

def general_response(state):
    messages = state["messages"]
    prompt = general_prompt.invoke({"messages": messages})
    response = model.invoke(prompt)
    return {"messages": [response]}
    

# Build graph
graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("rag_response", rag_response)
graph_builder.add_node("general_response", general_response)


# Logic 
graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges("retrieve", handle_response)
graph_builder.add_edge("rag_response", END)
graph_builder.add_edge("general_response", END)



memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def get_response_generator(query: str):
    input_messages = [HumanMessage(query)]
    config = {"configurable": {"thread_id": "abc123"}}
    
    try:
        for chunk in graph.stream(
            {"messages": input_messages}, 
            config,
            stream_mode="messages"
        ):
            if isinstance(chunk, tuple):
                message_chunk = chunk[0]
                if message_chunk.content:
                    yield f"data: {message_chunk.content}\n\n"
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"

@app.post("/chat")
async def chat_endpoint(query: str):
    return StreamingResponse(
        get_response_generator(query),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )

@app.get("/")
async def root():
    return {"message": "Thoughtful AI Support Agent is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)