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
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.output_connections: List[WebSocket] = []
    
    async def connect_input(self, websocket: WebSocket):
        """Connect an input client."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Input client connected. Total: {len(self.active_connections)}")
    
    async def connect_output(self, websocket: WebSocket):
        """Connect an output client."""
        await websocket.accept()
        self.output_connections.append(websocket)
        print(f"Output client connected. Total: {len(self.output_connections)}")
    
    def disconnect_input(self, websocket: WebSocket):
        """Disconnect an input client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Input client disconnected. Total: {len(self.active_connections)}")
    
    def disconnect_output(self, websocket: WebSocket):
        """Disconnect an output client."""
        if websocket in self.output_connections:
            self.output_connections.remove(websocket)
            print(f"Output client disconnected. Total: {len(self.output_connections)}")
    
    async def broadcast_to_outputs(self, message: dict):
        """Send message to all output clients."""
        if not self.output_connections:
            return
            
        disconnected = []
        for connection in self.output_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect_output(conn)


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
        """Handle streaming output from the LLM and broadcast to clients."""
        if not self.llm:
            return
            
        output_stream = self.llm.get_output_stream()
        
        async for token in output_stream:
            # Broadcast token to all output clients
            await self.manager.broadcast_to_outputs({
                "type": "token",
                "data": token
            })
            
            # Small delay to prevent overwhelming clients
            await asyncio.sleep(0.01)
    
    async def process_input(self, message: str):
        """Process input message through the LLM."""
        if not self.is_initialized:
            await self.initialize()
        
        print(f"Processing input: {message}")
        
        # Broadcast the user input to output clients
        await self.manager.broadcast_to_outputs({
            "type": "user_input",
            "data": message
        })
        
        # Send to LLM
        await self.llm.push_input(message)
    
    async def reset_cache(self):
        """Reset the LLM KV cache."""
        if self.llm:
            await self.llm.reset()
            await self.manager.broadcast_to_outputs({
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
        "input_clients": len(server.manager.active_connections),
        "output_clients": len(server.manager.output_connections)
    }


@app.websocket("/ws/input")
async def websocket_input_endpoint(websocket: WebSocket):
    """WebSocket endpoint for input clients."""
    await server.manager.connect_input(websocket)
    
    try:
        while True:
            # Receive message from input client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "message":
                await server.process_input(message_data["data"])
            elif message_data["type"] == "reset":
                await server.reset_cache()
            
    except WebSocketDisconnect:
        server.manager.disconnect_input(websocket)


@app.websocket("/ws/output")
async def websocket_output_endpoint(websocket: WebSocket):
    """WebSocket endpoint for output clients."""
    await server.manager.connect_output(websocket)
    
    try:
        # Keep connection alive
        while True:
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        server.manager.disconnect_output(websocket)
