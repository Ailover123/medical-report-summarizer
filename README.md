🧠 AI Medical Report Summarizer

This is a Streamlit web application that uses the Google Gemini API to summarize medical reports into simple, easy-to-understand language for patients.

Features

File Upload: Supports uploading medical reports in .pdf and .txt formats.

Text Input: Allows users to directly paste the text of a medical report.

AI-Powered Summarization: Utilizes the Gemini model to generate structured summaries that explain key findings, potential risks, and suggested next steps.

User-Friendly Interface: A clean and intuitive UI built with Streamlit.

Setup and Installation

1. Clone the repository

git clone <your-repository-url>
cd <your-repository-directory>


2. Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


3. Install dependencies

pip install -r requirements.txt


4. Set up your API Key

Create a file named .env in the root of your project directory.

Add your Google Gemini API key to the .env file like this:

GEMINI_API_KEY="YOUR_API_KEY_HERE"


5. Run the application

streamlit run app.py


The application should now be running in your web browser!