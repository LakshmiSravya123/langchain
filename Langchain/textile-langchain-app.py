import os
from dotenv import load_dotenv

# Import the chain and parser from our new textile_assistant file
from textile_assistant import chain, parser

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ====================================================================
# 3. Main function for the application
# ====================================================================
def main():
    """
    Main function to run the interactive textile assistant application.
    """
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
