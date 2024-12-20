from sentence_transformers import SentenceTransformer
import torch
import re
import chromadb
import pandas as pd

# -------------------------
# Text Preprocessing Function
# -------------------------
def preprocess_text(text):
    """
    Preprocesses the input text by converting it to lowercase,
    removing extra spaces, and stripping punctuation.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# -------------------------
# Sentence Splitting Function
# -------------------------
def split_into_sentences(text):
    """
    Splits the text into individual sentences using regular expressions.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# -------------------------
# Batch Processing Function
# -------------------------
def process_in_batches(data_list, batch_size):
    """
    Splits a list into smaller batches of specified size.
    """
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

# -------------------------
# Load CSV Data
# -------------------------
csv_file = "extracted_data.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Ensure the CSV has the required columns
if not {'URL', 'Title', 'Extracted Content'}.issubset(data.columns):
    raise ValueError("CSV must contain 'URL', 'Title', and 'Extracted Content' columns.")

# Extract content and corresponding URLs
content_list = data['Extracted Content'].dropna().tolist()  # Remove rows with NaN
urls = data['URL'].tolist()  # Store URLs for metadata

# -------------------------
# Preprocess and Split Sentences
# -------------------------
all_sentences = []
sentence_urls = []

for content, url in zip(content_list, urls):
    sentences = split_into_sentences(content)  # Split content into sentences
    all_sentences.extend(sentences)  # Collect all sentences
    sentence_urls.extend([url] * len(sentences))  # Duplicate the URL for each sentence

# Preprocess all sentences
preprocessed_sentences = [preprocess_text(sentence) for sentence in all_sentences]

# -------------------------
# Load Model and Generate Embeddings
# -------------------------
# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(preprocessed_sentences, convert_to_tensor=True)

# Normalize embeddings
normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

# Convert embeddings to a list of lists for compatibility with Chroma
embedding_list = normalized_embeddings.tolist()

# -------------------------
# Initialize Chroma Client
# -------------------------
client = chromadb.PersistentClient(path="db")

# Create or access a collection
collection = client.get_or_create_collection("sentence_embeddings_collection")

# -------------------------
# Add Embeddings to Database in Batches
# -------------------------
# Define maximum batch size
max_batch_size = 5461

# Prepare batches of sentences, URLs, and embeddings
sentence_batches = list(process_in_batches(all_sentences, max_batch_size))
url_batches = list(process_in_batches(sentence_urls, max_batch_size))
embedding_batches = list(process_in_batches(embedding_list, max_batch_size))

# Add each batch to the Chroma collection
for i, (sentence_batch, url_batch, embedding_batch) in enumerate(zip(sentence_batches, url_batches, embedding_batches)):
    # Create unique IDs for the batch
    batch_ids = [f"id_{i}_{j}" for j in range(len(sentence_batch))]
    
    # Add the batch to the database
    collection.add(
        ids=batch_ids,
        embeddings=embedding_batch,
        metadatas=[
            {"sentence": sentence, "url": url}
            for sentence, url in zip(sentence_batch, url_batch)
        ]
    )
    print(f"Batch {i + 1}/{len(sentence_batches)} added successfully.")

print("Database preparation completed.")