import os
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Montgo-Talk API", description="Backend for the City of Montgomery AI Assistant")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

BRIGHT_DATA_API_TOKEN = os.getenv("BRIGHT_DATA_API_TOKEN")

# Ingestion function: Read CSV using pandas
def load_sanitation_data(filepath="sanitation_data.csv"):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, filepath)
        
        if not os.path.exists(full_path):
            return "No local sanitation data found."
            
        df = pd.read_csv(full_path)
        
        data_string = "City of Montgomery Data:\n"
        for _, row in df.iterrows():
            # Create a clean string format
            row_dict = row.to_dict()
            details = " | ".join([f"{k}: {v}" for k, v in row_dict.items()])
            data_string += f"- {details}\n"
            
        return data_string
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return "Local data ingestion failed."

# Scraping function: Bright Data Web Unlocker
def fetch_live_alerts():
    if not BRIGHT_DATA_API_TOKEN:
        return "Live alerts unavailable (missing Bright Data API Token)."
        
    url = "https://api.brightdata.com/request"
    headers = {
        "Authorization": f"Bearer {BRIGHT_DATA_API_TOKEN}",
        "Content-Type": "application/json"
    }
    # Payload targeting the specified URL
    payload = {
        "zone": "web_unlocker", 
        "url": "https://www.montgomeryal.gov",
        "format": "raw"
    }
    
    try:
        # Wrap in try/except so the app doesn't crash
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        
        html_content = response.text
        
        # Simple extraction logic
        if "alert" in html_content.lower():
            return "Live Scraped Notice: The City of Montgomery homepage is currently accessible and may have active alerts."
        else:
            return "No active homepage alerts found during scraping."
            
    except Exception as e:
        print(f"Scraping API error: {e}")
        return "Live website data currently unavailable due to scraping error."

# Chat API Model
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# System Prompt
SYSTEM_PROMPT = """You are Montgo-Talk, an official Montgomery city assistant. Answer the user's question using ONLY the provided CSV data and live website data. Keep answers simple and citizen-friendly. If the answer is not in the provided data, you MUST say: 'I do not have enough information in the current city documents to answer that. Please contact the City of Montgomery directly.' Do not hallucinate."""

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set.")
        
    # Gather dynamic context data
    csv_context = load_sanitation_data()
    live_context = fetch_live_alerts()
    
    # Combine user question with context
    combined_prompt = (
        f"INSTRUCTIONS: {SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT START ---\n"
        f"CSV Data Context:\n{csv_context}\n\n"
        f"Live Web Scrape Context:\n{live_context}\n"
        f"--- CONTEXT END ---\n\n"
        f"User Question: {request.message}"
    )
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        chat_response = model.generate_content(combined_prompt)
        
        return ChatResponse(response=chat_response.text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to communicate with AI model. API Error: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Montgo-Talk FastAPI Backend is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
