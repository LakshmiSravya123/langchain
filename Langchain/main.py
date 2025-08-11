import os
import textwrap
from dotenv import load_dotenv
from typing import List, Dict

# LangChain Imports
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ====================================================================
# 1. Define the Textile Product Dataset
# ====================================================================
# In a real-world application, this data would be loaded from a CSV,
# database, or API. For this example, we'll keep it simple in a list of dicts.
TEXTILE_DATASET = [
    {
        "product_id": "T101",
        "name": "Organic Cotton T-Shirt",
        "material": "100% Organic Cotton",
        "color": "Natural Beige",
        "available_sizes": ["S", "M", "L", "XL"],
        "price": 25.00,
        "description": "A classic, soft t-shirt made from eco-friendly organic cotton. Breathable and comfortable for everyday wear.",
        "care_instructions": "Machine wash cold, tumble dry low."
    },
    {
        "product_id": "T102",
        "name": "Bamboo Blend Lounge Pants",
        "material": "70% Bamboo Viscose, 30% Spandex",
        "color": "Charcoal Gray",
        "available_sizes": ["S", "M", "L"],
        "price": 45.00,
        "description": "Luxuriously soft and stretchy lounge pants, perfect for relaxing at home. The bamboo viscose provides a silky feel.",
        "care_instructions": "Hand wash or machine wash on a delicate cycle. Do not bleach."
    },
    {
        "product_id": "T103",
        "name": "Recycled Polyester Fleece Jacket",
        "material": "100% Recycled Polyester",
        "color": "Forest Green",
        "available_sizes": ["M", "L", "XL", "XXL"],
        "price": 75.00,
        "description": "A warm and durable fleece jacket made entirely from recycled plastic bottles. Features a full-zip front and two side pockets.",
        "care_instructions": "Machine wash warm. Hang to dry."
    },
    {
        "product_id": "T104",
        "name": "Linen Button-Up Shirt",
        "material": "100% Linen",
        "color": "Sky Blue",
        "available_sizes": ["S", "M", "XL"],
        "price": 60.00,
        "description": "A lightweight and breathable linen shirt, ideal for warm weather. It has a relaxed fit and a classic collar.",
        "care_instructions": "Machine wash cold. Iron on a low setting."
    }
]

# ====================================================================
# 2. Define the Pydantic model for structured output
# ====================================================================
# This helps the LLM to return information in a specific, predictable format.
class ProductInfo(BaseModel):
    product_name: str = Field(description="The name of the textile product.")
    material: str = Field(description="The primary material of the product.")
    sizes_available: List[str] = Field(description="A list of sizes that are in stock.")
    price: float = Field(description="The price of the product in USD.")
    care_instructions: str = Field(description="Specific instructions for washing and drying the product.")

# ====================================================================
# 3. Initialize the LLM and RAG Pipeline
# ====================================================================
# Initialize the OpenAI model.
llm = OpenAI(api_key="open_AI_KEY", temperature=0)

# Create an output parser for our Pydantic model
parser = PydanticOutputParser(pydantic_object=ProductInfo)

# Define a prompt template that incorporates the dataset and the output format instructions
prompt_template = PromptTemplate(
    template=textwrap.dedent("""
    You are a helpful assistant for a textile company. Your task is to provide detailed product information based on the user's query.

    Use the following product data to answer the user's question. If the user asks about a product that doesn't exist in the data, state that you cannot find it.
    
    Product Data:
    {product_data}
    
    Format the output as JSON according to the following schema:
    {format_instructions}

    User Query: {query}
    Assistant:
    """),
    input_variables=["query"],
    partial_variables={"product_data": str(TEXTILE_DATASET), "format_instructions": parser.get_format_instructions()},
)

# Create the LangChain pipeline by chaining the prompt and the LLM
chain = prompt_template | llm

# ====================================================================
# 4. Main function for the application
# ====================================================================
def main():
    print("Welcome to the Textile Company Assistant!")
    print("Ask me about our products. Type 'exit' to quit.")
    
    while True:
        user_query = input("\n> ")
        if user_query.lower() == 'exit':
            break
        
        try:
            # Invoke the chain to get a response from the LLM
            response_text = chain.invoke({"query": user_query})
            
            # Use the parser to validate and parse the response
            parsed_response = parser.parse(response_text)
            
            # Print the formatted output
            print("\n--- Product Details ---")
            print(f"Product: {parsed_response.product_name}")
            print(f"Material: {parsed_response.material}")
            print(f"Available Sizes: {', '.join(parsed_response.sizes_available)}")
            print(f"Price: ${parsed_response.price:.2f}")
            print(f"Care: {parsed_response.care_instructions}")
            print("-----------------------")
        
        except Exception as e:
            # Handle cases where the LLM can't find a matching product or
            # returns an improperly formatted response.
            print("\nI'm sorry, I couldn't find a product that matches your query or process your request. Please try again.")
            print(f"Error: {e}")
            print("-----------------------")

if __name__ == "__main__":
    # Ensure the API key is set before running
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
    else:
        main()
