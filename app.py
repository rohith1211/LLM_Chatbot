import csv
import datetime
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import torch
import chromadb
import numpy as np
import google.generativeai as genai  # Import Google Gemini API
import uuid  # Import for unique ID generation

# Initialize Flask app
app = Flask(__name__)
app.chat_history = []
app.context = ""
app.chat_history_string = ""

# Initialize Chroma client and collection
client = chromadb.PersistentClient(path="db")
collection = client.get_collection("sentence_embeddings_collection")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up Gemini API Key (replace "your-gemini-api-key" with your actual API key)
genai.configure(api_key="AIzaSyBnqcQvTrdRZqGKw_ajWpMfvzda5EdIZFc")

# Helper function to retrieve context
def get_context(sentence_index, all_sentences, context_range=5):
    start = max(0, sentence_index - context_range)
    end = min(len(all_sentences), sentence_index + context_range + 1)
    return " ".join(all_sentences[start:end])

# Query function to interact with Chroma DB
def query_database(query, top_k=1, context_range=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    all_sentences = [metadata["sentence"] for metadata in collection.get()["metadatas"]]
    
    matched_sentences = []
    for result in results["metadatas"][0]:
        matched_sentence = result["sentence"]
        sentence_index = all_sentences.index(matched_sentence)
        context = get_context(sentence_index, all_sentences, context_range)
        matched_sentences.append({
            "matched_sentence": matched_sentence,
            "context": context,
            "url": result["url"]
        })
    return matched_sentences

# Function to generate AI response based on the context using Gemini API
def generate_ai_response(query, context, chat_history):
    prompt = "You are an AI helpdesk agent that is employed to answer queries about Changi Airport. \
More details about the content can be found in the provided context. \
Answer the questions to the point in a friendly and manner, make the user feel like he is welcome. \
If the answer is not present in the context given, then dont use your knowledge to answer the question, instead say the answer is not present in the context."

    if len(chat_history) > 0:
        prompt = prompt + '\n' + "This is the previous exchage you had with the user" + '\n' + chat_history

    prompt = prompt + '\n \n' + f"Context: {context}"

    prompt = prompt + '\n \n' + "Based on above instructions and context respond to the below query given by the user."

    prompt = prompt + '\n \n' + f"Query: {query}"

    prompt = prompt + "Your response: "
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Use the desired Gemini model
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating Gemini AI response: {e}")
        return "Unable to generate Gemini AI response at this time."

# Function to log prompts and responses to a CSV file
# def log_to_csv(query, context, prompt, response, is_gemini=False):
#     with open('ai_responses_log.csv', mode='a', newline='') as file:
#         writer = csv.writer(file)
#         current_time = datetime.datetime.now()
#         source = "Gemini" if is_gemini else "OpenAI"
#         row = [current_time.strftime("%Y-%m-%d %H:%M:%S"), str(uuid.uuid4()), query, context, prompt, response, source]
#         writer.writerow(row)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data.get('message')
    app.chat_history.append({"user": user_message})

    if user_message:
        # Query the database for matching context
        results = query_database(user_message, top_k=1, context_range=5)
        if results:
            matched_context = results[0]["context"]
            # Use Gemini API to generate AI response
            bot_response = generate_ai_response(user_message, matched_context, app.chat_history_string)
        else:
            bot_response = user_message  # Default response if no match found
        
        # Log data to CSV with Gemini indication
        # prompt = f"Query: {user_message} | Context: {matched_context} | Generate a detailed and coherent response..."
        # log_to_csv(user_message, matched_context, prompt, bot_response, is_gemini=True)
        
        app.chat_history_string += f"User: {user_message} \n"
        app.chat_history_string += f"Bot: {bot_response} \n"

        app.chat_history.append({'bot': bot_response, 'context': matched_context})
        return jsonify({"response": bot_response, "context": matched_context})
    
    return jsonify({"response": "No message received!"})

if __name__ == '__main__':
    app.run(debug=True)
