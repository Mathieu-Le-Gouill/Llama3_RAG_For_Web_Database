import requests
import os

# Ensure the directory exists
if not os.path.exists('pdf_data'):
    os.makedirs('pdf_data')

# Open the file and read the links
with open('PDFlinks.txt', 'r') as file:
    for line in file:
        url = line.strip()  # Remove newline characters
        response = requests.get(url)

        # Get the file name from the URL
        file_name = url.split("/")[-1]

        # Write the content to a file
        with open(os.path.join('pdf_data', file_name), 'wb') as pdf_file:
            pdf_file.write(response.content)