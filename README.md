![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/dev-llama3-blog-1920x1080-1.png)

# Llama 3.1 Chatbot for Questions & Answers

This project is a Streamlit-based chatbot that uses the Llama 3.1 language model to answer questions about the "The Llama 3 Herd of Models" paper. It employs a RAG (Retrieval-Augmented Generation) approach, allowing the chatbot to provide responses based on the content of the paper.

## Features

- Uses Llama 3.1 models (70B and 8B versions) for natural language processing
- Implements document embedding and retrieval using FAISS
- Allows customization of various parameters like chunk size, overlap, and top K documents to retrieve
- Provides a user-friendly interface for asking questions and viewing responses

## Setup Instructions

### 1. Set up the .env file

Create a file named `.env` in the root directory of the project and add the following content:

```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Replace `your_google_api_key_here` and `your_groq_api_key_here` with your actual API keys.

### 2. Install requirements

To install the required packages, run the following command in your terminal:

```
pip install -r requirements.txt
```

Make sure you have Python installed on your system before running this command.

### 3. Run the app

To run the Streamlit app, use the following command in your terminal:

```
streamlit run app.py
```

This will start the Streamlit server and open the app in your default web browser.

## Usage

1. When the app loads, use the sidebar to customize settings such as the model choice, chunk size, and top K documents to retrieve.
2. Click the "Documents Embedding" button to process the PDF file and create vector embeddings.
3. Once the embeddings are created, you can start asking questions in the text input field.
4. The chatbot will provide answers based on the content of "The Llama 3 Herd of Models" paper.
5. You can view the source documents used for each answer by expanding the "Document Similarity Search" section.

## Note

Ensure that the PDF file "The Llama 3 Herd of Models" is placed in the `Data` folder before running the app.

## Contributing

Feel free to fork this project and submit pull requests with any improvements or bug fixes.
