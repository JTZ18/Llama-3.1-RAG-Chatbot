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
# If running locally
OLLAMA_BASE_URL=http://localhost:11434

# If deploying as a kubernetes service on runai cluster
OLLAMA_BASE_URL=http://ollama.runai-aiap-17-ds.svc.cluster.local:80
```

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

## To Deploy on a Kubernetes Cluster

1. Build the Docker image:

```
docker build -t llama-3.1-rag-chatbot:latest .
```

2. Push the Docker image to a container registry:

```
docker tag llama-3.1-rag-chatbot:latest asia-southeast1-docker.pkg.dev/aiap-17-ds/aiap-17-ds/jon_taylor_lim_mt/llama-3.1-rag-chatbot:v1.1
docker push asia-southeast1-docker.pkg.dev/aiap-17-ds/aiap-17-ds/jon_taylor_lim_mt/llama-3.1-rag-chatbot:v1.1
```

3. Apply the Kubernetes configuration:
Ensure you run these commands on the kubernetes cluster you want to deploy to.
```
kubectl apply -f k8s/ragui.yaml
kubectl apply -f k8s/ollama.yaml
kubectl apply -f k8s/ingress.yaml
```

4. Verify the deployment:
Access the Streamlit app here: https://llama31-rag.aiap17.aisingapore.net/


## Contributing

Feel free to fork this project and submit pull requests with any improvements or bug fixes.
