## Textile LangChain Application
This project is a simple, command-line chatbot that acts as an assistant for a textile company. It uses the LangChain framework to implement a Retrieval-Augmented Generation (RAG) pipeline, allowing it to answer questions about a predefined set of textile products. The application is built to provide structured product information based on user queries by leveraging a Pydantic model for predictable output from the LLM.

## Project Structure
The application is modularized into three Python files:

 - _textile-langchain-app.py_ : The main entry point of the application. It handles user input and runs the interactive loop.

 - _textile_assistant.py_ : Contains the core RAG pipeline. This file defines the ProductInfo Pydantic model, sets up the LangChain prompt template, and creates the chain that connects the LLM with the product data.

 - _textile_data.py_ : A simple data file that holds a list of dictionaries representing the textile products. In a real-world scenario, this data would likely be sourced from a database or an external API.

## Setup and Installation
### Prerequisites
 - Python 3.6 or higher

 - An OpenAI API key

1. Install Dependencies
Install the necessary Python packages using pip. You will need langchain, langchain-openai, and python-dotenv.
```python
pip install langchain langchain-openai python-dotenv
```

2. Configure Your API Key
Create a file named .env in the root directory of your project (in the same folder as the Python files). Add your OpenAI API key to this file in the following format:
```python
OPENAI_API_KEY="your_openai_api_key_here"
```

Note: The .env file should not be committed to Git, as it contains a secret. It's a good practice to add .env to your .gitignore file.

### How to Run
To start the application, simply run the main Python file from your terminal:
```python
python textile-langchain-app.py
```

The application will start, and you can begin asking questions about the textile products. To exit the application, type exit.

### How it Works
The application's logic is a basic RAG flow:

- Retrieve: The user's query is combined with a list of all product data in the prompt.

- Augment: The LLM is instructed to use this product data as its source of truth.

- Generate: The LLM generates a response in a structured JSON format, as defined by the ProductInfo Pydantic model.

- Parse: The PydanticOutputParser validates the LLM's JSON response and converts it into a Python object, which is then neatly printed to the console.
