"""
Live LLM Server - FastAPI server with WebSocket support for live streaming
"""
import asyncio
import json
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from live_llm import GemmaLiveLLM


class ConnectionManager:
    """Manages a single WebSocket connection."""
    
    def __init__(self):
        self.connection: WebSocket | None = None
    
    async def connect(self, websocket: WebSocket):
        """Connect the client. Replace existing connection if present."""
        # Close existing connection if present
        if self.connection is not None:
            try:
                await self.connection.close(code=1000, reason="Replaced by new connection")
            except:
                pass  # Connection might already be closed
            print("Replacing existing connection")
            
        await websocket.accept()
        self.connection = websocket
        print("Client connected")
        return True
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect the client."""
        if websocket == self.connection:
            self.connection = None
            print("Client disconnected")
    
    async def send_message(self, message: dict):
        """Send message to the client."""
        if self.connection is None:
            return
            
        try:
            await self.connection.send_text(json.dumps(message))
        except:
            # Connection is broken, clean it up
            self.connection = None
    
    def is_connected(self) -> bool:
        """Check if a client is connected."""
        return self.connection is not None


class LiveLLMServer:
    """Live LLM Server managing the model and connections."""
    
    def __init__(self):
        self.llm: GemmaLiveLLM = None
        self.manager = ConnectionManager()
        self.is_initialized = False
        self.output_task = None
    
    async def initialize(self):
        """Initialize the LLM model."""
        if self.is_initialized:
            return
            
        print("Initializing Gemma 3 270M model...")
        self.llm = GemmaLiveLLM("google/gemma-3-270m-it")
        await self.llm.initialize_model()
        
        # Start output streaming task
        self.output_task = asyncio.create_task(self._output_stream_handler())
        
        self.is_initialized = True
        print("✓ Model initialized and ready!")
    
    async def _output_stream_handler(self):
        """Handle streaming output from the LLM and send to the client."""
        if not self.llm:
            return
            
        output_stream = self.llm.get_output_stream()
        
        async for token in output_stream:
            # Check for end-of-response signal
            if token == "[END_OF_RESPONSE]":
                # Send end signal to client
                await self.manager.send_message({
                    "type": "end",
                    "data": ""
                })
            else:
                # Send token to client
                await self.manager.send_message({
                    "type": "token",
                    "data": token
                })
            
            # Small delay to prevent overwhelming client
            await asyncio.sleep(0.01)
    
    async def process_input(self, message: str):
        """Process input message through the LLM."""
        if not self.is_initialized:
            await self.initialize()
        
        print(f"Processing input: {message}")
        
        # Send the user input to client (for logging/debugging)
        await self.manager.send_message({
            "type": "user_input",
            "data": message
        })
        
        # Send to LLM
        await self.llm.push_input(message)
    
    async def reset_cache(self):
        """Reset the LLM KV cache."""
        if self.llm:
            await self.llm.reset()
            await self.manager.send_message({
                "type": "system",
                "data": "✓ KV cache reset"
            })


# Global server instance
server = LiveLLMServer()

# FastAPI app
app = FastAPI(title="Live LLM Server", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    await server.initialize()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "Live LLM Server is running", "initialized": server.is_initialized}


@app.get("/status")
async def status():
    """Get server status."""
    return {
        "initialized": server.is_initialized,
        "client_connected": server.manager.is_connected()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Single WebSocket endpoint for bidirectional communication."""
    connected = await server.manager.connect(websocket)
    if not connected:
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "message":
                await server.process_input(message_data["data"])
            elif message_data["type"] == "reset":
                await server.reset_cache()
            
    except WebSocketDisconnect:
        server.manager.disconnect(websocket)
