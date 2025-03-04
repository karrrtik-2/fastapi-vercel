from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import time

from pymongo import MongoClient
import os
from groq import Groq
from dotenv import load_dotenv
from functools import lru_cache
import re

# Load environment variables
load_dotenv()

# Initialize global clients
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongodb_client['ChatBot']
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Global variables
conversation_history = []
final_conversation_history = []  # New: For second Groq LLM - only original user inputs and raw responses

# In-memory data cache
parent_data_cache = {}
children_data_cache = {}
data_loaded = False

# Define FastAPI app
app = FastAPI(title="Medical Chatbot API", description="API for medical product recommendations", version="1.0.0")

# Pydantic models for request and response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    processing_time: float

def load_all_data():
    """Load all needed data into memory once"""
    global parent_data_cache, children_data_cache, data_loaded
    
    if data_loaded:
        return
    
    # Fetch all parent documents with projection to reduce memory usage
    parent_collection = db['parent']
    parents = list(parent_collection.find({}, {
        "_id": 0, 
        "Parent_id": 1, 
        "Category": 1, 
        "Medical Features": 1, 
        "Tags": 1, 
        "Nutritional Info": 1
    }))
    
    # Index parents by ID for fast lookup
    for parent in parents:
        parent_id = parent["Parent_id"]
        parent_data_cache[parent_id] = parent
    
    # Fetch all children documents with needed fields
    children_collection = db['children']
    children = list(children_collection.find({}, {"_id": 0}))
    
    # Group children by parent_id
    for child in children:
        parent_id = child.get("Parent_id")
        if parent_id:
            if parent_id not in children_data_cache:
                children_data_cache[parent_id] = []
            children_data_cache[parent_id].append(child)
    
    data_loaded = True

@lru_cache(maxsize=1)
def read_system_message(file_path):
    """Cache the system message to avoid repeated file reads"""
    start_time = time.time()
    try:
        with open(file_path, 'r') as file:
            result = file.read().strip()
    except FileNotFoundError:
        result = "Default system message: I'm a helpful assistant."
    end_time = time.time()
    return result

def parse_values(text):
    """Parse comma-separated values"""
    return [value.strip() for value in text.split(',')]

def filter_parents_in_memory(assistant_response):
    matching_parent_ids = []
    category_values = []
    medical_features = []
    tags = []
    nutritional_info = []
    
    # Extract search criteria from assistant response
    if "Category:" in assistant_response:
        category_text = assistant_response.split("Category:")[1].strip().split('\n')[0]
        category_values = parse_values(category_text)
    
    if "Medical Features:" in assistant_response:
        medical_text = assistant_response.split("Medical Features:")[1].strip().split('\n')[0]
        medical_features = parse_values(medical_text)
    
    if "Tags:" in assistant_response:
        tags_text = assistant_response.split("Tags:")[1].strip().split('\n')[0]
        tags = parse_values(tags_text)
    
    if "Nutritional Info:" in assistant_response:
        nutritional_text = assistant_response.split("Nutritional Info:")[1].strip().split('\n')[0]
        nutritional_info = parse_values(nutritional_text)
    
    # Filter parents based on criteria
    for parent_id, parent in parent_data_cache.items():
        # Check if parent matches any category
        category_match = False
        if category_values:
            parent_category = parent.get("Category", "")
            for category in category_values:
                if category.lower() in parent_category.lower():
                    category_match = True
                    break
        else:
            category_match = True  # No category filter specified
        
        if not category_match:
            continue
        
        # Check for additional criteria
        other_match = True
        
        # Check medical features
        if medical_features and other_match:
            parent_medical = parent.get("Medical Features", "")
            med_match = False
            for feature in medical_features:
                if feature.lower() in parent_medical.lower():
                    med_match = True
                    break
            other_match = med_match
        
        # Check tags
        if tags and other_match:
            parent_tags = parent.get("Tags", "")
            tags_match = False
            for tag in tags:
                if tag.lower() in parent_tags.lower():
                    tags_match = True
                    break
            other_match = tags_match
        
        # Check nutritional info
        if nutritional_info and other_match:
            parent_nutrition = parent.get("Nutritional Info", "")
            nutrition_match = False
            for info in nutritional_info:
                if info.lower() in parent_nutrition.lower():
                    nutrition_match = True
                    break
            other_match = nutrition_match
        
        # If parent matches all criteria, add to results
        if category_match and other_match:
            matching_parent_ids.append(parent_id)
            
            # Limit to 10 parents
            if len(matching_parent_ids) >= 10:
                break
    
    return matching_parent_ids

def get_children_for_parents(parent_ids, limit=10):
    
    children_data = []
    link_map = {}  # Dictionary to store Link -> Link_value mapping
    
    for parent_id in parent_ids:
        if parent_id in children_data_cache:
            children_data.extend(children_data_cache[parent_id])
            
            # Limit total children
            if len(children_data) >= limit:
                children_data = children_data[:limit]
                break
    
    # Clean the data (remove images, links) but save the link values for later
    cleaned_children_data = []
    for item in children_data:
        # Save the link value before removing it
        if "Link" in item and "Link_value" in item:
            link_map[item["Link"]] = item["Link_value"]
        
        # Make a copy to avoid modifying the original
        cleaned_item = item.copy()
        
        # Explicitly remove any "Images" key if it exists
        if "Images" in cleaned_item:
            del cleaned_item["Images"]
        
        # Explicitly remove the "Link_value" key if it exists
        if "Link_value" in cleaned_item:
            del cleaned_item["Link_value"]
        
        # Look for any fields containing image URLs
        for key in list(cleaned_item.keys()):
            if isinstance(cleaned_item[key], str) and (
                "http" in cleaned_item[key] and any(ext in cleaned_item[key].lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])
            ):
                # If the field appears to be solely an image URL, remove it
                if key.lower() in ['image', 'images', 'img', 'thumbnail', 'photo']:
                    del cleaned_item[key]
        
        cleaned_children_data.append(cleaned_item)
    return cleaned_children_data, link_map

def replace_link_placeholders(response_text, link_map):
    """Replace [Link-X] placeholders with actual link values, and handle entire response being Link-X"""
    # Check if the entire response is Link-X
    entire_link_pattern = r'^Link-(\d+)$'
    entire_match = re.match(entire_link_pattern, response_text.strip())
    if entire_match:
        link_id = f"Link-{entire_match.group(1)}"
        if link_id in link_map:
            return link_map[link_id]
        else:
            return response_text  # If link not found, return original
    
    # Otherwise, replace [Link-X] placeholders within the text
    link_pattern = r'\[Link-(\d+)\]'
    matches = re.findall(link_pattern, response_text)
    
    modified_response = response_text
    for match in matches:
        link_id = f"Link-{match}"
        placeholder = f"[{link_id}]"
        if link_id in link_map:
            actual_link = link_map[link_id]
            modified_response = modified_response.replace(placeholder, actual_link)
    
    return modified_response

def process_with_groq(user_input, system_message):
    """Process user input with Groq model and return response"""
    total_start_time = time.time()
    try:
        global conversation_history, final_conversation_history
        
        # Ensure data is loaded
        if not data_loaded:
            load_all_data()
        if not system_message:
            system_message = read_system_message("keys.txt")
        
        # Add user input to both conversation histories
        conversation_history.append({"role": "user", "content": user_input})
        final_conversation_history.append({"role": "user", "content": user_input})
        
        initial_messages = [
            {"role": "system", "content": system_message}
        ] + conversation_history
        
        # First API call to get category information
        groq_first_call_start = time.time()
        response = groq_client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=initial_messages,
            temperature=0.0,
            max_tokens=1024
        )
        groq_first_call_end = time.time()
        print(f"Time for first Groq API call: {groq_first_call_end - groq_first_call_start:.4f} seconds")
        
        assistant_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Process query in memory
        query_process_start = time.time()
        matching_parent_ids = filter_parents_in_memory(assistant_response)
        query_process_end = time.time()
        print(f"Time to process query in memory: {query_process_end - query_process_start:.4f} seconds")
        
        # Get children data from memory and link map
        children_data, link_map = get_children_for_parents(matching_parent_ids) if matching_parent_ids else ([], {})
        
        enhanced_system_message = """You are a helpful medical assistant that provides product suggestions based on the available products data but if no data is provided then answer from your side DONT USE below FORMAT.
            Only use "₹" sign for prices.
            If user talks in Hindi, respond in Hindi but in English script.
            Do not create your own links, just provide link from the data.
            Recommend min 1 and max 3 products from the provided data(if provided otherwise dont recommend any product).
            Response should be to the point and concise, also don't mention unnecessary info or comments.
            Please analyze the product information and provide clear recommendations in a proper format including:
            1. Product names
               - Prices
               - Sizes
               - [Link]
               
            Available Product Data:
            """
        
        # Prepare product data string efficiently
        product_data_str = "\n".join(str(child) for child in children_data)
        enhanced_system_message += product_data_str
        
        modified_user_input = f"{user_input} (response should be in proper format and easy to read. Respond like you're assisting)"
        conversation_history.append({"role": "user", "content": modified_user_input})

        # Use final_conversation_history for the second API call
        final_messages = [
            {"role": "system", "content": enhanced_system_message}
        ] + final_conversation_history
        
        print(f"here si the messages:{final_messages}")
        # Second API call with product information
        groq_second_call_start = time.time()
        final_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-specdec",
            messages=final_messages,
            temperature=0.0,
            max_tokens=1024
        )
        groq_second_call_end = time.time()
        print(f"Time for second Groq API call: {groq_second_call_end - groq_second_call_start:.4f} seconds")
        
        l_assistant_response = final_response.choices[0].message.content
        
        # Replace link placeholders with actual link values
        final_assistant_response = replace_link_placeholders(l_assistant_response, link_map)
        
        conversation_history.append({"role": "assistant", "content": final_assistant_response})
        # Add the raw LLM response to final_conversation_history
        final_conversation_history.append({"role": "assistant", "content": l_assistant_response})

        total_end_time = time.time()
        print(f"Total processing time: {total_end_time - total_start_time:.4f} seconds")
        return final_assistant_response, total_end_time - total_start_time
    
    except Exception as e:
        total_end_time = time.time()
        print(f"Error occurred. Total processing time: {total_end_time - total_start_time:.4f} seconds")
        raise Exception(f"Error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        system_message = read_system_message("keys.txt")
        response, processing_time = process_with_groq(request.message, system_message)
        return ChatResponse(response=response, processing_time=processing_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_conversation():
    """Reset the conversation history"""
    global conversation_history, final_conversation_history
    conversation_history = []
    final_conversation_history = []
    return {"status": "conversation reset successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "data_loaded": data_loaded}

# On startup event to load data
@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    load_all_data()
    print("Data loaded successfully on startup")

if __name__ == "__main__":
    # Start FastAPI with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
