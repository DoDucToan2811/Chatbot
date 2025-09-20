import datetime
import os
import openai
import nest_asyncio
from pathlib import Path
from pdfminer.high_level import extract_text
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from chainlit.context import get_context
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.agent.openai import OpenAIAgent
from uuid import uuid4
import time
import chainlit as cl
from literalai import LiteralClient
import requests  
from collections import deque
from typing import Dict, Optional
from chainlit.types import ThreadDict

# Load API Keys from environment
os.environ["OPENAI_API_KEY"] = "your_api_key_go_here"
LITERAL_API_KEY = os.getenv("LITERAL_API_KEY")
os.environ["OAUTH_GOOGLE_CLIENT_ID"] = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
os.environ["OAUTH_GOOGLE_CLIENT_SECRET"] = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")
print(os.environ["OAUTH_GOOGLE_CLIENT_ID"])
print(os.environ["OAUTH_GOOGLE_CLIENT_SECRET"])
# Set your OpenAI key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize the Literal AI client
lai = LiteralClient(api_key=LITERAL_API_KEY)
lai.instrument_openai()

# Apply nest_asyncio to avoid issues with async loops in Jupyter and other environments
nest_asyncio.apply()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
pdf_files = {
    2022: Path("./BOSCH/bosch_2022.pdf"),
    2021: Path("./BOSCH/bosch_2021.pdf"),
    2020: Path("./BOSCH/bosch_2020.pdf"),
    2019: Path("./BOSCH/bosch_2019.pdf"),
    2018: Path("./BOSCH/bosch_2018.pdf"),
    2017: Path("./BOSCH/bosch_2017.pdf"),
    2016: Path("./BOSCH/bosch_2016.pdf"),
    2015: Path("./BOSCH/bosch_2015.pdf"),
    2014: Path("./BOSCH/bosch_2014.pdf"),
    2013: Path("./BOSCH/bosch_2013.pdf"),
    2012: Path("./BOSCH/bosch_2012.pdf"),
    2011: Path("./BOSCH/bosch_2011.pdf"),
    2010: Path("./BOSCH/bosch_2010.pdf"),
}

# Extract Text from PDFs
def read_pdf(file_path):
    try:
        return extract_text(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Load and Process PDFs
doc_set = {}
for year, pdf_path in pdf_files.items():
    pdf_content = read_pdf(pdf_path)
    if pdf_content:
        doc_set[year] = [{"content": pdf_content, "metadata": {"year": year}}]

# Create Vector Index
index_set = {}
storage_base_dir = "./storage/bosch"
for year, docs in doc_set.items():
    storage_context = StorageContext.from_defaults()
    documents = [Document(text=doc["content"]) for doc in docs]
    cur_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index_set[year] = cur_index
    storage_context.persist(persist_dir=f"{storage_base_dir}/{year}")

# Load Index for Querying
for year in doc_set.keys():
    storage_context = StorageContext.from_defaults(persist_dir=f"{storage_base_dir}/{year}")
    index_set[year] = load_index_from_storage(storage_context)

# Initialize Query Engine Tools
individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[year].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{year}",
            description=f"Useful for answering queries about the {year} Bosch annual report.",
        ),
    )
    for year in index_set.keys()
]

# Define SubQuestionQueryEngine
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    llm=OpenAI(model="gpt-4o-mini"),
)

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="Useful for answering queries about the {year} Bosch annual report.",
    ),
)

# Debug query tools
print("Query Engine Tools Debug:")
for tool in individual_query_engine_tools:
    print(f"Tool Name: {tool.metadata.name}, Description: {tool.metadata.description}")

def initialize_chatbot_for_years(memory):
    try:
        agent = OpenAIAgent.from_tools(
            tools=[query_engine_tool],
            memory=memory
        )
        return agent
    except Exception as e:
        print(f"Error initializing OpenAIAgent: {e}")
        return None

# Chainlit Integration
memory = deque(maxlen=50)  # To store the conversation history

@cl.on_chat_start
async def initialize_chat_session():
    await cl.Message(
        author="assistant",
        content="Hello! I'm an AI assistant. How may I help you?"
    ).send()
    
@cl.on_message
async def handle_user_message(message: cl.Message):
    # Retrieve the chat store path and memory from the user session
    history_path = cl.user_session.get("history_path")
    memory = cl.user_session.get("memory")

    # Initialize context to get thread_id for unique identification
    context = get_context()
    thread_id = context.session.thread_id

    # If memory or history_path is None, initialize them
    if history_path is None or memory is None:
        history_path = Path(f"./history/{thread_id}.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        chat_store = SimpleChatStore()
        cl.user_session.set("history_path", str(history_path))
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=chat_store,
            chat_store_key=thread_id
        )
        cl.user_session.set("memory", memory)

    # Extract the content of the user's message
    message_content = message.content

    # Initialize an agent
    agent = initialize_chatbot_for_years(memory)
    if not agent:
        await cl.Message(content="Failed to initialize chatbot. Please try again later.").send()
        return

    # Generate a response from the assistant
    try:
        response = agent.chat(message_content)

        # Check if the response is a dictionary or contains additional data
        if isinstance(response, str):
            response_content = response
            print("Case1",)
        elif hasattr(response, "content"):
            response_content = response.content
            print("Case2")
        else:
            response_content = str(response)
            print("Case3")
    except Exception as e:
        print(f"Error during chat generation: {e}")
        response_content = "Sorry, I encountered an error while processing your request."

    # Persist the updated chat store to the history file
    if memory.chat_store:
        memory.chat_store.persist(str(history_path))

    # Send the assistant's response back to the user
    await cl.Message(content=response).send()

@cl.on_chat_resume
async def resume_chat_session(thread: ThreadDict):
    history_path = cl.user_session.get("history_path")
    history_path = Path(history_path)
    chat_store = SimpleChatStore.from_persist_path(str(history_path))
    context = get_context()
    thread_id = context.session.thread_id
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key=thread_id
    )
    cl.user_session.set("memory", memory)

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User
) -> Optional[cl.User]:
    """Handle OAuth callback."""
    if provider_id == "google":
        user_email = raw_user_data.get("email")
        if user_email:
            return cl.User(identifier=user_email, metadata={"role": "user"})
    return None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    cl.run(host="0.0.0.0", port=port)
