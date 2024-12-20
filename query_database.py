from sentence_transformers import SentenceTransformer
import torch
import chromadb
import numpy as np

# -------------------------
# Helper Function to Retrieve Context
# -------------------------
def get_context(sentence_index, all_sentences, context_range=5):
    """
    Retrieves the context around a matched sentence.
    :param sentence_index: Index of the matched sentence in all_sentences
    :param all_sentences: List of all sentences in the database
    :param context_range: Number of sentences to include before and after
    :return: Contextual sentences as a single string
    """
    start = max(0, sentence_index - context_range)  # Ensure start index is not negative
    end = min(len(all_sentences), sentence_index + context_range + 1)  # Ensure end index is within bounds
    return " ".join(all_sentences[start:end])

# -------------------------
# Initialize Chroma Client and Collection
# -------------------------
client = chromadb.PersistentClient(path="db")

# Access the collection
collection = client.get_collection("sentence_embeddings_collection")

# Retrieve all sentences and URLs from the database for context
collection_data = collection.get()
all_sentences = [metadata["sentence"] for metadata in collection_data["metadatas"]]
all_urls = [metadata["url"] for metadata in collection_data["metadatas"]]

# -------------------------
# Load Model
# -------------------------
# Use the same model as used for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Query the Database
# -------------------------
def query_database(query, top_k=1, context_range=5):
    """
    Searches the Chroma database for the most relevant sentences.
    :param query: User's search query
    :param top_k: Number of top results to retrieve
    :param context_range: Number of sentences to include before and after the match
    :return: Matched sentence and its context
    """
    # Preprocess the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0).tolist()

    # Perform a nearest neighbor search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Parse the results
    matched_sentences = []
    for result in results["metadatas"][0]:
        matched_sentence = result["sentence"]
        sentence_index = all_sentences.index(matched_sentence)  # Find the sentence index
        context = get_context(sentence_index, all_sentences, context_range)
        matched_sentences.append({
            "matched_sentence": matched_sentence,
            "context": context,
            "url": result["url"]
        })

    return matched_sentences

# -------------------------
# Main Functionality
# -------------------------
if __name__ == "__main__":
    # Prompt user for query input
    user_query = input("Enter your query: ")

    # Retrieve results from the database
    results = query_database(user_query, top_k=1, context_range=5)

    # Display results
    for result in results:
        print("\nMatched Sentence:")
        print(result["matched_sentence"])
        print("\nContext:")
        print(result["context"])
        print("\nSource URL:")
        print(result["url"])