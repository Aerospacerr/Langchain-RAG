from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    """
    Returns the embedding function to be used for vectorization.
    Currently uses OllamaEmbeddings with the 'llama3' model.
    """

    # Using OllamaEmbeddings with the 'llama3' model
    embeddings = OllamaEmbeddings(model="llama3")
    return embeddings
