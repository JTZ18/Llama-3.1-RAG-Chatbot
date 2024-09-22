import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Import HTML templates for chat messages
from htmlTemplates import css, bot_template, user_template

# Load environment variables from .env file
load_dotenv()

# Load API keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Function for document embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings_initialized = False
        st.sidebar.write("Initializing embeddings...")

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        st.sidebar.write("Embeddings initialized.")

        st.sidebar.write("Loading documents from PDF directory...")
        st.session_state.loader = PyPDFDirectoryLoader("./Data")  # Data Ingestion from PDF folder
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.sidebar.write(f"{len(st.session_state.docs)} documents loaded.")

        st.sidebar.write("Splitting documents into chunks...")
        chunk_size = st.session_state.chunk_size
        chunk_overlap = st.session_state.chunk_overlap
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.sidebar.write(f"{len(st.session_state.final_documents)} chunks created.")

        st.sidebar.write("Creating vector embeddings...")
        
        # Process documents in batches of 100
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(st.session_state.final_documents), batch_size):
            batch = st.session_state.final_documents[i:i+batch_size]
            texts = [doc.page_content for doc in batch]
            try:
                batch_embeddings = st.session_state.embeddings.embed_documents(texts)
                all_embeddings.extend(batch_embeddings)
                st.sidebar.write(f"Processed {min(i+batch_size, len(st.session_state.final_documents))} out of {len(st.session_state.final_documents)} documents")
            except Exception as e:
                st.sidebar.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                raise

        # Create FAISS index from all embeddings
        import numpy as np
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        texts = [doc.page_content for doc in st.session_state.final_documents]
        metadatas = [doc.metadata for doc in st.session_state.final_documents]
        
        st.session_state.vectors = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_array)),
            embedding=st.session_state.embeddings,
            metadatas=metadatas
        )

        st.sidebar.write("Vector embeddings created.")
        st.session_state.embeddings_initialized = True
        
        # Reinitialize the conversation chain with the new vectorstore
        st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectors, st.session_state.model_choice)

# Function to get the conversational retrieval chain
def get_conversation_chain(vectorstore, model_name):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        max_length=st.session_state.conversational_memory_length, 
        output_key='answer'
        )
    
    # Create a retriever with the specified top_k
    retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.top_k})
    
   
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True  # Ensure source documents are returned
    )
    return conversation_chain

# Add customization options to the sidebar
st.sidebar.title('Customization')
model = st.sidebar.selectbox(
    'Choose a model',
    ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant'],
    key='model_choice',  # Unique key for the selectbox
)


st.session_state.conversational_memory_length = st.sidebar.slider('Conversational memory length:', 0, 10, value=0)
st.session_state.chunk_size = st.sidebar.slider('Chunk size:', 1000, 8000, value=1000, step=500)
st.session_state.chunk_overlap = st.sidebar.slider('Chunk overlap:', 0, 1000, value=100, step=100)
st.session_state.top_k = st.sidebar.slider('Top K documents to retrieve:', 1, 20, value=20)

# Add a text area for the prompt
prompt = st.sidebar.text_area("Enter a prompt for the LLM:", key="prompt", value ="You are an AI assistant with deep expertise in the Llama 3 foundation model. Your role is to explain the model's architecture, its training process, and capabilities following training. Users may inquire about topics such as pre-training data, model scaling, or performance evaluation experiments. Provide clear and insightful responses based on the document provided. If any requested information is not covered in the document, you may draw on your own knowledge, but ensure the user is aware that this information is not directly from the given source.")

if st.sidebar.button("Documents Embedding"):
    vector_embedding()
    st.sidebar.write("Vector Store DB Is Ready")
    st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectors, model)  # Initialize the conversation chain

# Displaying a GIF
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://cdn.dribbble.com/users/488656/screenshots/14380421/llamadrama.gif" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Llama 3.1 Chatbot for Questions & Answers")

# Session state variable
if 'chat_history' not in st.session_state:
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
    st.session_state.conversation_chain.retriever.search_kwargs["k"] = st.session_state.top_k
    
    # Invoke the retrieval chain with the prompt included
    response = st.session_state.conversation_chain({
        'question': f"{prompt} {user_question}", 
        'chat_history': st.session_state.chat_history
        })
    ai_response = response['answer']

    # Save the user input and AI response to the chat history
    st.session_state.chat_history.append({'human': user_question, 'AI': ai_response})

    st.session_state.user_question = ""

    display_chat_history()

    # Store response for document similarity search
    st.session_state.response = response

def display_chat_history():
    st.write(css, unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", message['human']), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message['AI']), unsafe_allow_html=True)

def main():
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    st.text_input("Ask Llama 3.1 any question about 'The Llama 3 Herd of models' paper:", key="user_question", on_change=handle_userinput)

    # With a streamlit expander
    if 'response' in st.session_state:
        with st.expander("Document Similarity Search"):
            if "source_documents" in st.session_state.response:
                for i, doc in enumerate(st.session_state.response["source_documents"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No source documents found in the response.")

if __name__ == '__main__':
    main()