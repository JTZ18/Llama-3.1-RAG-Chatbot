import streamlit as st
import os
import time
import numpy as np
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from pathlib import Path

# Import HTML templates for chat messages
from htmlTemplates import css, bot_template, user_template

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")


# Function for document embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings_initialized = False
        st.sidebar.write("Initializing embeddings...")

        st.session_state.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model="nomic-embed-text",
        )

        # Define the path for the FAISS index
        faiss_index_path = Path("./faiss_index")

        # Always load documents and create chunks
        st.sidebar.write("Loading documents from PDF directory...")
        loader = PyPDFDirectoryLoader("./Data")  # Data Ingestion from PDF folder
        docs = loader.load()  # Document Loading
        st.sidebar.write(f"{len(docs)} documents loaded.")

        st.sidebar.write("Splitting documents into chunks...")
        chunk_size = st.session_state.chunk_size
        chunk_overlap = st.session_state.chunk_overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )  # Chunk Creation
        final_documents = text_splitter.split_documents(docs)  # Splitting
        st.sidebar.write(f"{len(final_documents)} chunks created.")

        # Check if the FAISS index already exists
        if faiss_index_path.exists():
            st.sidebar.write("Loading existing FAISS index...")
            st.session_state.vectors = FAISS.load_local(
                str(faiss_index_path),
                st.session_state.embeddings,
                allow_dangerous_deserialization=True,
            )
            st.sidebar.write("FAISS index loaded.")
        else:
            st.sidebar.write("Creating new FAISS index...")

            # Process documents in batches of 100
            batch_size = 100
            all_embeddings = []
            for i in range(0, len(final_documents), batch_size):
                batch = final_documents[i : i + batch_size]
                texts = [doc.page_content for doc in batch]
                try:
                    batch_embeddings = st.session_state.embeddings.embed_documents(
                        texts
                    )
                    all_embeddings.extend(batch_embeddings)
                    st.sidebar.write(
                        f"Processed {min(i+batch_size, len(final_documents))} out of {len(final_documents)} documents"
                    )
                except Exception as e:
                    st.sidebar.error(
                        f"Error processing batch {i//batch_size + 1}: {str(e)}"
                    )
                    raise

            # Create FAISS index from all embeddings
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            texts = [doc.page_content for doc in final_documents]
            metadatas = [doc.metadata for doc in final_documents]

            st.session_state.vectors = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings_array)),
                embedding=st.session_state.embeddings,
                metadatas=metadatas,
            )

            # Save the FAISS index
            st.session_state.vectors.save_local(str(faiss_index_path))
            st.sidebar.write("FAISS index created and saved.")

        st.session_state.embeddings_initialized = True

        # Reinitialize the conversation chain with the new vectorstore
        st.session_state.conversation_chain = get_conversation_chain(
            st.session_state.vectors, st.session_state.model_choice
        )


# Function to get the conversational retrieval chain
def get_conversation_chain(vectorstore, model_name):
    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_length=st.session_state.conversational_memory_length,
        output_key="answer",
    )

    # Create a retriever with the specified top_k
    retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.top_k})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # Ensure source documents are returned
    )
    return conversation_chain


# Add customization options to the sidebar
st.sidebar.title("Customization")
model = st.sidebar.selectbox(
    "Choose a model",
    ["finalend/hermes-3-llama-3.1"],
    key="model_choice",  # Unique key for the selectbox
)


st.session_state.conversational_memory_length = st.sidebar.slider(
    "Conversational memory length:", 0, 10, value=0
)
st.session_state.chunk_size = st.sidebar.slider(
    "Chunk size:", 1000, 8000, value=1000, step=500
)
st.session_state.chunk_overlap = st.sidebar.slider(
    "Chunk overlap:", 0, 1000, value=100, step=100
)
st.session_state.top_k = st.sidebar.slider(
    "Top K documents to retrieve:", 1, 20, value=3
)

# Add a text area for the prompt
prompt = st.sidebar.text_area(
    "Enter a prompt for the LLM:",
    key="prompt",
    value="You are an AI assistant with deep expertise in the Llama 3 foundation model. Your role is to explain the model's architecture, its training process, and capabilities following training. Users may inquire about topics such as pre-training data, model scaling, or performance evaluation experiments. Provide clear and insightful responses based on the document provided. If any requested information is not covered in the document, you may draw on your own knowledge, but ensure the user is aware that this information is not directly from the given source.",
)

if st.sidebar.button("Documents Embedding"):
    vector_embedding()
    st.sidebar.write("Vector Store DB Is Ready")
    st.session_state.conversation_chain = get_conversation_chain(
        st.session_state.vectors, model
    )  # Initialize the conversation chain

# Add a button to force re-embedding if needed
if st.sidebar.button("Force Re-embedding"):
    if os.path.exists("./faiss_index"):
        import shutil

        shutil.rmtree("./faiss_index")
    st.session_state.pop("vectors", None)
    st.session_state.embeddings_initialized = False
    st.sidebar.write(
        "Existing embeddings cleared. Click 'Documents Embedding' to create new embeddings."
    )

# Displaying a GIF
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://cdn.dribbble.com/users/488656/screenshots/14380421/llamadrama.gif" width="150">
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Llama 3.1 Chatbot for Questions & Answers")

# Session state variable
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def handle_userinput():
    user_question = st.session_state.user_question
    if not st.session_state.get("embeddings_initialized", False):
        st.write("Please initialize the embeddings first.")
        return

    if "conversation_chain" not in st.session_state:
        st.write("Please initialize the conversation chain first.")
        return

    start = time.process_time()

    # Update the retriever with the current top_k value
    st.session_state.conversation_chain.retriever.search_kwargs["k"] = (
        st.session_state.top_k
    )

    # Invoke the retrieval chain with the prompt included
    response = st.session_state.conversation_chain(
        {
            "question": f"{prompt} {user_question}",
            "chat_history": st.session_state.chat_history,
        }
    )
    ai_response = response["answer"]

    # Save the user input and AI response to the chat history
    st.session_state.chat_history.append({"human": user_question, "AI": ai_response})

    st.session_state.user_question = ""

    display_chat_history()

    # Store response for document similarity search
    st.session_state.response = response


def display_chat_history():
    st.write(css, unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        st.write(
            user_template.replace("{{MSG}}", message["human"]), unsafe_allow_html=True
        )
        st.write(bot_template.replace("{{MSG}}", message["AI"]), unsafe_allow_html=True)


def main():
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    st.text_input(
        "Ask Llama 3.1 any question about 'The Llama 3 Herd of models' paper:",
        key="user_question",
        on_change=handle_userinput,
    )

    # With a streamlit expander
    if "response" in st.session_state:
        with st.expander("Document Similarity Search"):
            if "source_documents" in st.session_state.response:
                for i, doc in enumerate(st.session_state.response["source_documents"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No source documents found in the response.")


if __name__ == "__main__":
    main()