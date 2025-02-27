from pymongo import MongoClient
import os
from groq import Groq
from dotenv import load_dotenv
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app - minimalist version
app = FastAPI()

# Initialize global clients directly at module level
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongodb_client['ChatBot']  # Get database reference once
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Global variables initialized once
conversation_history = []

@lru_cache(maxsize=1)
def read_system_message(file_path):
    """Cache the system message to avoid repeated file reads"""
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return "Default system message: I'm a helpful assistant."

def get_children_data(collection, parent_ids):
    """Get children data for the given parent IDs"""
    # Use projection to only fetch needed fields
    children_data = []
    
    # Batch find with a single query instead of loop
    if parent_ids:
        children_docs = collection.find({"Parent_id": {"$in": parent_ids}}, {"_id": 0})
        children_data = list(children_docs)
    
    return children_data

def parse_values(text):
    """Parse comma-separated values"""
    return [value.strip() for value in text.split(',')]

def build_query(assistant_response):
    """Extract query information from assistant response"""
    # Initialize category query and other conditions
    category_query = {}
    other_conditions = []
    
    if "Category:" in assistant_response:
        category_text = assistant_response.split("Category:")[1].strip().split('\n')[0]
        category_values = parse_values(category_text)
        category_conditions = []
        for value in category_values:
            category_conditions.append({
                "Category": {
                    "$regex": value,
                    "$options": "i"
                }
            })
        category_query = {"$or": category_conditions} if len(category_conditions) > 1 else category_conditions[0]
    
    fields = ["Medical Features:", "Tags:", "Nutritional Info:"]
    for field in fields:
        if field in assistant_response:
            field_text = assistant_response.split(field)[1].strip().split('\n')[0]
            field_values = parse_values(field_text)
            
            field_conditions = []
            for value in field_values:
                field_conditions.append({
                    field.rstrip(":"): {
                        "$regex": value,
                        "$options": "i"
                    }
                })
            other_conditions.extend(field_conditions)
    
    final_query = category_query
    if other_conditions:
        if category_query:
            final_query = {
                "$and": [
                    category_query,
                    {"$or": other_conditions}
                ]
            }
        else:
            final_query = {"$or": other_conditions}
    
    return final_query

def process_with_groq(user_input, system_message):
    """Process user input with Groq model and return response"""
    try:
        global conversation_history
        
        parent_collection = db['parent']
        children_collection = db['children']
        
        if not system_message:
            system_message = read_system_message("keys.txt")
        
        conversation_history.append({"role": "user", "content": user_input})
        initial_messages = [
            {"role": "system", "content": system_message}
        ] + conversation_history
        
        # First API call to get category information
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=initial_messages,
            temperature=0.0,
            max_tokens=1024
        )
        
        assistant_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Build MongoDB query
        final_query = build_query(assistant_response)
        
        if final_query:
            # Use projection to only get necessary fields
            matching_docs = parent_collection.find(final_query, {"Parent_id": 1, "_id": 0})
            parent_ids = [doc["Parent_id"] for doc in matching_docs]
            
            if parent_ids:
                # Get children data in one query instead of multiple
                projection = {"_id": 0, "Images": 0}
                children_docs = children_collection.find({"Parent_id": {"$in": parent_ids}}, projection)
                children_data = list(children_docs)
                
                cleaned_children_data = []
                for item in children_data:
                    # Make a copy to avoid modifying the original
                    cleaned_item = item.copy()
                    
                    # Explicitly remove any "Images" key if it exists
                    if "Images" in cleaned_item:
                        del cleaned_item["Images"]
                    
                    # Look for any fields containing image URLs
                    for key in list(cleaned_item.keys()):
                        if isinstance(cleaned_item[key], str) and (
                            "http" in cleaned_item[key] and 
                            any(ext in cleaned_item[key].lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])
                        ):
                            # If the field appears to be solely an image URL, remove it
                            if key.lower() in ['image', 'images', 'img', 'thumbnail', 'photo']:
                                del cleaned_item[key]
                    
                    cleaned_children_data.append(cleaned_item)
                
                # Prepare system message with product data
                enhanced_system_message = """
                You are a helpful medical assistant that provides product suggestions based on the available products data.
                Only use "₹" sign for prices.\n
                If user talks in Hindi, respond in Hindi but in English script.\n
                Do not generate your own links or add anything to the existing links, just provide link from the data.\n
                Recommend minimum 1 product, maximum 3 product.\n
                Response should be to the point and concise, also don't mention unnecessary info or comments.\n
                Please analyze the product information and provide clear recommendations including:
                1. Product names
                2. Prices
                3. Sizes
                Format the response in a clear, easy-to-read manner.
                Available Product Data:
                """
                
                # Prepare product data string efficiently using CLEANED data
                product_data_str = "\n".join(str(child) for child in cleaned_children_data)
                enhanced_system_message += product_data_str
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": user_input})

                final_messages = [
                    {"role": "system", "content": enhanced_system_message}
                ] + conversation_history
                
                # Second API call with product information
                final_response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=final_messages,
                    temperature=0.0,
                    max_tokens=1024
                )
                
                final_assistant_response = final_response.choices[0].message.content
                conversation_history.append({"role": "assistant", "content": final_assistant_response})

                return final_assistant_response
            else:
                return "No matching products found."
        
        return assistant_response

    except Exception as e:
        raise Exception(f"Error: {str(e)}")

# Define Pydantic models for API
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Only the /chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    try:
        system_message = read_system_message("keys.txt")
        response = process_with_groq(chat_message.message, system_message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CLI interface (for testing)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ChatBot as a FastAPI server or CLI")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of API server")
    args = parser.parse_args()
    
    if args.cli:
        # CLI mode
        while True:
            user_input = input("\nEnter your message (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
            
            try:
                system_message = read_system_message("keys.txt")
                result = process_with_groq(user_input, system_message)
                print(result)
            except Exception as e:
                print(f"Error: {str(e)}")
    else:
        # API server mode
        print("Starting FastAPI server at http://127.0.0.1:8000")
        print("Only /chat endpoint is available")
        uvicorn.run(app, host="127.0.0.1", port=8000)
