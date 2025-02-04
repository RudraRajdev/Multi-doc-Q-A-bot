from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class DocumentRetriever:
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initialize the DocumentRetriever with a transformer model.

        Args:
        - model_name (str): Name of the transformer model to use for embeddings.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, documents):
        """
        Create embeddings for a list of documents.

        Args:
        - documents (list of str): List of document contents.

        Returns:
        - np.array: Array of document embeddings.
        """
        embeddings = []
        for doc in documents:
            inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use the mean of the token embeddings as the document embedding
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)

    def embed_query(self, query):
        """
        Create an embedding for a user query.

        Args:
        - query (str): The user's question.

        Returns:
        - np.array: The query embedding.
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def retrieve_documents(question, documents):
    """
    Retrieve relevant documents based on the user's question.

    Args:
    - question (str): The user's question.
    - documents (list of str): List of document contents.

    Returns:
    - list of str: List of relevant documents based on the question.
    """
    retriever = DocumentRetriever()

    # Create embeddings for the documents and the question
    document_embeddings = retriever.embed_documents(documents)
    question_embedding = retriever.embed_query(question)

    # Simple retrieval logic (e.g., cosine similarity)
    relevant_documents = []
    for idx, doc_embedding in enumerate(document_embeddings):
        similarity = calculate_similarity(question_embedding, doc_embedding)
        if similarity > 0.5:  # Adjust the threshold as needed
            relevant_documents.append(documents[idx])

    return relevant_documents

def calculate_similarity(vec_a, vec_b):
    """
    Calculate similarity between the query and document embeddings.

    Args:
    - vec_a (array): The embedding of the query.
    - vec_b (array): The embedding of the document.

    Returns:
    - float: Similarity score.
    """
    return cosine_similarity(vec_a, vec_b)

def cosine_similarity(vec_a, vec_b):
    """
    Calculate the cosine similarity between two vectors.

    Args:
    - vec_a (array): First vector.
    - vec_b (array): Second vector.

    Returns:
    - float: Cosine similarity score.
    """
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
