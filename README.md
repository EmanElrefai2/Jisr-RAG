# Project: Jisr RAG


## Overview
A chatbot capable of answering users’ questions based on the knowledge extracted from PDF documents. The application should allow users to upload documents and chat with the chatbot using a simple, accessible interface.


## Documentation

For detailed documentation, please visit the 


***

## Python Version

   > python 3.10

***

 ## env variables
You can access and download the env variables from [here]()

***

## Installation

1. Clone the repository: `git clone [<repository_url>](https://github.com/.git)`
2. Navigate to the project directory: `cd `
3. Create the environment: `python3.10 -m venv myenv`
4. Activate the environment: `source myenv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Run the main application: `python main.py`


## Usage

1. Run the main application: `python main.py`
2. Access the FastAPI endpoints by navigating to `http://{machine_ip}:8080` in your browser or public link.


## Docker

To run the application using Docker:

1. Build the Docker image: `docker build -t gradio-app .`
2. Run the Docker container: `docker run -p 7860:7860 gradio-app`
3. Access the application using the appropriate IP address or domain name and port 7860.




