from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chat Webhook API", version="1.0.0")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# n8n webhook URL
WEBHOOK_URL = ""

class ChatRequest(BaseModel):
    chat_input: str
    session_id: str

class ChatResponse(BaseModel):
    success: bool
    message: str
    webhook_response: Optional[dict] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that forwards data to n8n webhook
    
    Args:
        request: ChatRequest containing chat_input and session_id
        
    Returns:
        ChatResponse with success status and webhook response
    """
    try:
        # Prepare data to send to webhook
        webhook_data = {
            "chat_input": request.chat_input,
            "session_id": request.session_id
        }
        
        logger.info(f"Sending data to webhook for session: {request.session_id}")
        
        # Send POST request to n8n webhook (15 second timeout)
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                WEBHOOK_URL,
                json=webhook_data,
                headers={"Content-Type": "application/json"}
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            # Try to parse JSON response, fallback to text if not JSON
            try:
                webhook_response = response.json()
            except:
                webhook_response = {"response": response.text}
            
            logger.info(f"Webhook responded with status: {response.status_code}")
            
            return ChatResponse(
                success=True,
                message="Webhook response received successfully",
                webhook_response=webhook_response
            )
            
    except httpx.TimeoutException:
        logger.error("Webhook request timed out after 15 seconds")
        raise HTTPException(
            status_code=408,
            detail="Webhook request timed out after 15 seconds"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Webhook returned error status: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Webhook error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Chat Webhook API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "webhook_url": WEBHOOK_URL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)