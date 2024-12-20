import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
import datetime
import subprocess

# Base URL of the website
base_url = "https://www.changiairport.com/"
visited_urls = set()

# File path for visited URLs
urls_file = f"visited_urls.csv"

# Initialize CSV file
with open(urls_file, "w", newline='', encoding='utf-8') as uf:
    csv_writer = csv.writer(uf)
    csv_writer.writerow(["Visited URLs"])

def scrape_page(url):
    if url in visited_urls:
        return
    visited_urls.add(url)

    # Save the URL to the CSV file
    with open(urls_file, "a", newline='', encoding='utf-8') as uf:
        csv_writer = csv.writer(uf)
        csv_writer.writerow([url])

    # Fetch the page
    try:
        response = requests.get(url)
        if response.status_code != 200:
            # Skip URLs that fail to fetch
            return
    except requests.RequestException:
        # Catch any request exceptions and skip the URL
        return

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")
    print(f"Visited: {url}")

    # Find and scrape subpage links
    links = soup.find_all("a", href=True)
    for link in links:
        sub_url = urljoin(base_url, link["href"])
        if base_url in sub_url:  # Stay within the website
            scrape_page(sub_url)

scrape_page(base_url)

print(f"URLs extraction is succcessfully completed. Results saved in {urls_file}.")

# Call extract_data.py after the print statement
subprocess.run(["python", "extract_data.py"])