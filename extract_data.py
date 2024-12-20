import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import time

# File path containing the URLs
input_csv_file = "visited_urls.csv"  # Change to your file's name
output_file = "extracted_data.csv"

# Read URLs from the CSV file
df = pd.read_csv(input_csv_file)
urls = df['Visited URLs'].tolist()

# Custom User-Agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
}

# Function to fetch and extract data from a URL
def fetch_data_from_url(url):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Example of extracting specific data, modify as needed
            title = soup.title.string if soup.title else "No Title"
            paragraphs = [p.get_text() for p in soup.find_all('p')]  # Extract all paragraphs
            
            return title, paragraphs
        else:
            print(f"Failed to fetch {url} - Status code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None, None

# Open the output file for saving the data
with open(output_file, "w", newline='', encoding='utf-8') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["URL", "Title", "Extracted Content"])  # Write header row
    
    print("Starting data extraction...")
    
    for url in urls:
        print(f"Processing URL: {url}")
        title, paragraphs = fetch_data_from_url(url)
        
        # Introduce a delay to avoid rate limiting
        time.sleep(2)  # Delay 2 seconds between requests
        
        if title and paragraphs:
            csv_writer.writerow([url, title, ' '.join(paragraphs)])  # Save the data to the CSV
            print(f"Successfully extracted data from {url}")
        else:
            csv_writer.writerow([url, "Failed to extract data", ""])
            print(f"Failed to extract data from {url}")
    
    print(f"Data extraction completed. Results saved in {output_file}.")