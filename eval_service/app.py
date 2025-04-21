import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import json

from config import ServerConfig, ModelConfig, ToolConfig
from model_service import ModelService

# Define API request models
class ChatMessage(BaseModel):
    role: str  # Role of the message (system, user, assistant)
    content: str  # Content of the message

class ChatCompletionRequest(BaseModel):
    model: str  # Model identifier
    messages: List[ChatMessage]  # List of conversation messages
    temperature: Optional[float] = 0.7  # Controls randomness in generation
    top_p: Optional[float] = 0.9  # Nucleus sampling parameter
    max_tokens: Optional[int] = 1024  # Maximum number of tokens to generate

def create_app(config: ServerConfig):
    """
    Create and configure the FastAPI application
    
    Args:
        config: Server configuration object
        
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="LLM Code Tool Service",
        description="Large language model code tool calling service compatible with OpenAI API",
        version="1.0.0"
    )
    
    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize the model service
    model_service = ModelService(config.llm_config, config.tool_config)
    
    # Store service in application state
    app.state.model_service = model_service
    
    @app.on_event("startup")
    async def startup_event():
        """Load model when application starts"""
        app.state.model_service.load_model()
        print(f"Model loaded: {config.llm_config.model_path}")
    
    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, req: Request):
        """
        Chat completion API endpoint compatible with OpenAI
        
        Processes chat messages and returns model-generated responses
        with tool calling capabilities
        """
        try:
            # Convert Pydantic model to dictionary
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Generate response using the model service
            response = app.state.model_service.generate_response(messages)
            
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint to verify service availability"""
        return {"status": "healthy"}
    
    return app

if __name__ == "__main__":
    # Load configuration from environment variables
    config = ServerConfig(
        llm_config=ModelConfig(
            model_path=os.environ.get("MODEL_PATH", "yi-34b-chat"),  # Path to the model
        ),
        tool_config=ToolConfig(
            tool_server_url=os.environ.get("TOOL_SERVER_URL", "http://localhost:30094/get_observation"),  # URL for tool server
            valid_actions=json.loads(os.environ.get("VALID_ACTIONS", '["python"]')),  # List of valid tool actions
            max_turns=int(os.environ.get("MAX_TURNS", "5")),  # Maximum number of tool interaction turns
        ),
        host=os.environ.get("HOST", "0.0.0.0"),  # Host to bind the server
        port=int(os.environ.get("PORT", "8000")),  # Port to bind the server
    )
    
    # Create and run the application
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)