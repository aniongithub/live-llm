#!/usr/bin/env python3
"""
Live LLM Input Client - Simple CLI for sending messages to the LLM server
"""
import asyncio
import json
import websockets
import sys
import argparse
import time


class InputClient:
    """Simple input client for Live LLM server."""
    
    def __init__(self, host: str = "localhost", port: int = 8000, max_retries: int = 10):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.server_url = f"ws://{host}:{port}/ws/input"
        self.websocket = None
        self.running = True
    
    async def connect(self):
        """Connect to the server with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"Connecting to Live LLM server at {self.host}:{self.port} (attempt {attempt}/{self.max_retries})...")
                self.websocket = await websockets.connect(self.server_url)
                print("✓ Connected to Live LLM server")
                print("Type your messages and press Enter. Type 'quit' to exit, 'reset' to clear cache.")
                print("-" * 60)
                return
            except Exception as e:
                print(f"❌ Connection attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print("❌ Failed to connect after all retries")
                    print("Make sure the Live LLM server is running")
                    sys.exit(1)
    
    async def send_message(self, message: str):
        """Send a message to the server."""
        if not self.websocket:
            return
            
        try:
            if message.lower() == 'reset':
                await self.websocket.send(json.dumps({
                    "type": "reset"
                }))
                print("🔄 Cache reset requested")
            else:
                await self.websocket.send(json.dumps({
                    "type": "message",
                    "data": message
                }))
                print(f"📤 Sent: {message}")
        except Exception as e:
            print(f"❌ Error sending message: {e}")
    
    async def run(self):
        """Run the input client."""
        await self.connect()
        
        try:
            while self.running:
                # Get user input
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, input, "> "
                    )
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        self.running = False
                        break
                    
                    if user_input.strip():
                        await self.send_message(user_input.strip())
                        
                except KeyboardInterrupt:
                    self.running = False
                    break
                except EOFError:
                    self.running = False
                    break
        
        finally:
            if self.websocket:
                await self.websocket.close()
            print("\n👋 Goodbye!")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live LLM Input Client")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--max-retries", type=int, default=10, help="Maximum connection retries (default: 10)")
    
    args = parser.parse_args()
    
    client = InputClient(host=args.host, port=args.port, max_retries=args.max_retries)
    await client.run()


if __name__ == "__main__":
    print("🚀 Live LLM Input Client")
    print("=" * 30)
    asyncio.run(main())
