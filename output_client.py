#!/usr/bin/env python3
"""
Live LLM Output Client - Real-time display of LLM responses
"""
import asyncio
import json
import websockets
import sys
import os
import argparse


class OutputClient:
    """Output client for displaying Live LLM responses."""
    
    def __init__(self, host: str = "localhost", port: int = 8000, max_retries: int = 10):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.server_url = f"ws://{host}:{port}/ws/output"
        self.websocket = None
        self.running = True
        self.current_response = ""
        self.line_count = 0
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """Print the header."""
        print("🤖 Live LLM Output Stream")
        print("=" * 50)
        print()
    
    def update_display(self):
        """Update the display with current content."""
        self.clear_screen()
        self.print_header()
        
        if self.current_response:
            print("AI:", self.current_response)
        else:
            print("Waiting for responses...")
        
        print("\n" + "-" * 50)
        print("Press Ctrl+C to exit")
    
    async def connect(self):
        """Connect to the server with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"Connecting to Live LLM server at {self.host}:{self.port} (attempt {attempt}/{self.max_retries})...")
                self.websocket = await websockets.connect(self.server_url)
                print("✓ Connected to Live LLM server output stream")
                self.update_display()
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
    
    async def handle_message(self, message_data: dict):
        """Handle incoming message from server."""
        msg_type = message_data.get("type")
        data = message_data.get("data", "")
        
        if msg_type == "user_input":
            # New user input - start fresh
            self.current_response = ""
            # self.clear_screen()
            self.print_header()
            print(f"You: {data}")
            print("AI: ", end="", flush=True)
            
        elif msg_type == "token":
            # New token from AI
            self.current_response += data
            print(data, end="", flush=True)
            
        elif msg_type == "system":
            # System message
            # self.clear_screen()
            self.print_header()
            print(f"System: {data}")
            print("\nWaiting for responses...")
            print("\n" + "-" * 50)
            print("Press Ctrl+C to exit")
    
    async def run(self):
        """Run the output client."""
        await self.connect()
        
        try:
            async for message in self.websocket:
                try:
                    message_data = json.loads(message)
                    await self.handle_message(message_data)
                except json.JSONDecodeError:
                    print(f"❌ Invalid message format: {message}")
                except Exception as e:
                    print(f"❌ Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("\n❌ Connection to server lost")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        finally:
            if self.websocket:
                await self.websocket.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live LLM Output Client")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--max-retries", type=int, default=10, help="Maximum connection retries (default: 10)")
    
    args = parser.parse_args()
    
    client = OutputClient(host=args.host, port=args.port, max_retries=args.max_retries)
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
