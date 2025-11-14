"""
FastAPI HTTP Server for IPython Kernel Manager

This server exposes the IPython kernel manager functions as HTTP endpoints.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
import logging
import uvicorn

# Import the kernel manager functions
from verl_tool.servers.tools.utils.ipython_tool import (
    call_python_script_with_ipython_async,
    remove_kernel,
    get_kernel_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IPython Kernel Manager API",
    description="HTTP API for managing and executing Python scripts in isolated IPython kernels",
    version="1.0.0"
)

# Request/Response Models
class ExecuteScriptRequest(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    script: str = Field(..., description="Python script to execute")
    timeout: int = Field(default=120, ge=1, le=3600, description="Execution timeout in seconds")
    max_output_size: int = Field(default=100*1024, ge=1024, le=10*1024*1024, description="Maximum output size in bytes")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "user123_session456",
                "script": "print('Hello, World!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')",
                "timeout": 120,
                "max_output_size": 102400
            }
        }


class ExecuteScriptResponse(BaseModel):
    output: str = Field(..., description="Script output (stdout, stderr, or error messages)")
    success: bool = Field(..., description="Whether the script executed successfully")
    request_id: str = Field(..., description="The request_id that was executed")


class RemoveKernelRequest(BaseModel):
    request_id: str = Field(..., description="Request ID of the kernel to remove")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "user123_session456"
            }
        }


class RemoveKernelResponse(BaseModel):
    message: str = Field(..., description="Status message")
    request_id: str = Field(..., description="The request_id that was removed")


class StatsResponse(BaseModel):
    active_kernels: int = Field(..., description="Number of active kernels")
    max_kernels: int = Field(..., description="Maximum number of kernels allowed")
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")
    kernel_ids: list[str] = Field(..., description="List of active kernel IDs")


# API Endpoints
@app.post(
    "/execute",
    response_model=ExecuteScriptResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Python Script",
    description="Execute a Python script in an isolated IPython kernel"
)
async def execute_script(request: ExecuteScriptRequest):
    """
    Execute a Python script using IPython kernel.
    
    - **request_id**: Unique identifier for maintaining kernel state across requests
    - **script**: Python code to execute
    - **timeout**: Maximum execution time (default: 120 seconds)
    - **max_output_size**: Maximum size of output in bytes (default: 100KB)
    """
    try:
        logger.info(f"Executing script for request_id: {request.request_id}")
        
        output, success = await call_python_script_with_ipython_async(
            request_id=request.request_id,
            script=request.script,
            timeout=request.timeout,
            max_output_size=request.max_output_size
        )
        
        logger.info(f"Execution completed for {request.request_id}: success={success}")
        
        return ExecuteScriptResponse(
            output=output,
            success=success,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"Error executing script for {request.request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute script: {str(e)}"
        )


@app.delete(
    "/kernel/{request_id}",
    response_model=RemoveKernelResponse,
    status_code=status.HTTP_200_OK,
    summary="Remove Kernel",
    description="Remove a specific kernel and free its resources"
)
async def delete_kernel(request_id: str):
    """
    Remove a kernel by its request_id and clean up associated resources.
    
    - **request_id**: The identifier of the kernel to remove
    """
    try:
        logger.info(f"Removing kernel: {request_id}")
        remove_kernel(request_id)
        
        return RemoveKernelResponse(
            message=f"Kernel removed successfully",
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Error removing kernel {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove kernel: {str(e)}"
        )


@app.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Kernel Statistics",
    description="Get current statistics about kernel usage and memory"
)
async def get_stats():
    """
    Get current kernel manager statistics including active kernels and memory usage.
    """
    try:
        stats = get_kernel_stats()
        logger.info(f"Stats requested: {stats}")
        
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check if the server is running"
)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "ipython-kernel-manager"}


# Root endpoint
@app.get("/", summary="API Information")
async def root():
    """Get basic API information."""
    return {
        "name": "IPython Kernel Manager API",
        "version": "1.0.0",
        "endpoints": {
            "execute": "POST /execute - Execute Python scripts",
            "remove_kernel": "DELETE /kernel/{request_id} - Remove a kernel",
            "stats": "GET /stats - Get kernel statistics",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API documentation"
        }
    }


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
        reload: Enable auto-reload for development (default: False)
    """
    logger.info(f"Starting IPython Kernel Manager API server on {host}:{port}")
    uvicorn.run(
        "ipython_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IPython Kernel Manager HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    start_server(host=args.host, port=args.port, reload=args.reload)