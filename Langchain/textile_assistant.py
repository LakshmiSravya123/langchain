# ====================================================================
# This file contains the LangChain RAG pipeline for the textile assistant.
# It defines the Pydantic model, initializes the LLM, and creates the chain.
# ====================================================================

import os
import textwrap
from typing import List
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# Import the dataset from our new file
from textile_data import TEXTILE_DATASET

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ====================================================================
# 1. Define the Pydantic model for structured output
# ====================================================================
# This helps the LLM to return information in a specific, predictable format.
class ProductInfo(BaseModel):
    product_name: str = Field(description="The name of the textile product.")
    material: str = Field(description="The primary material of the product.")
    sizes_available: List[str] = Field(description="A list of sizes that are in stock.")
    price: float = Field(description="The price of the product in USD.")
    care_instructions: str = Field(description="Specific instructions for washing and drying the product.")

# ====================================================================
# 2. Initialize the LLM and RAG Pipeline
# ====================================================================
# Initialize the OpenAI model.
llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)

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
