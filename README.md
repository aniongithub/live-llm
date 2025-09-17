# Live LLM

An experimental LLM runtime that supports **live KV cache streaming** for conversational AI.  
Instead of re-sending the entire prompt and conversation history at every step, the client only sends **incremental state updates** that keep the model’s KV cache *hot*.  

This allows the LLM to respond faster and more interactively, at the expense of eventually forgetting the oldest parts of a conversation. Think of it as a **rolling window of conversational memory** that trades completeness for latency.

![Live LLM Demo](img/live-llm-demo.gif)

This is an actual demo of Gemma 270m running on CPU, with instantaneous, stateful responses since the KV cache is prepopulated with prompt and context.

---

## Use Cases

1. **Real-time Speech / TTS Agents**  
   - When paired with ASR + TTS, the hot cache enables fast, incremental responses  
   - Useful for low-latency voice assistants on edge devices
   - Can reset or swap KV cache to "forget" or switch to new conversational context

2. **Task oriented, assistant-type LLMs**  
   - Chatbots and assistants that need *snappy* replies within a given task without extensive context
   - Interactive storytelling or character responses

3. **Prototyping & Research**  
   - Great for experimenting with **streaming inference** and **incremental context**  

---

## Why Live KV Cache?

Traditional LLM runtimes rebuild the whole context on every request, which is expensive and slow.  
With a persistent **key-value cache**, each new message only needs to process the **delta** (the new tokens), which gives:

-  **Low Latency Responses** — No need to re-encode all prior messages  
-  **Lower Compute Cost** — Especially helpful on CPU-bound inference  
-  **Interactive Streaming** — Feels more like a real conversation than a batch job  
-  **Composable** — Clients can decide how much history to keep vs. reset  

---


## Advantages

-  **Speed**: Incremental generation is much faster on constrained hardware 
-  **Memory Efficiency**: Keeps just enough conversation history, instead of rebuilding context  
-  **Simple Integration**: WebSocket-based clients, easy to drop into other projects  
-  **Small-Model Friendly**: Runs with less then 1s to first-token with Gemma 3 270M on CPU (no GPU required)  
-  **Resettable**: Cache can be cleared mid-conversation for a fresh state  
-  **Context switchable**: Cache can be swapped mid-conversation for a new conversational goal

---

## Disadvantages / Trade-offs

-  **Single-Tenant**: Designed for *one conversation at a time* (this is why we use small models that can scale with each ongoing conversation)
-  **Forgets Old Context**: Since the cache is finite, earlier conversation turns are eventually dropped  
-  **Cache Invalidation Bugs**: Care must be taken to avoid mismatched inputs vs. cached states  
-  **Less Accurate Long-Form Reasoning**: Without full history, answers to very long threads may drift  

---

## Why This Repo Is Useful

- Demonstrates **how to wire up a live KV cache** in practice  
- Provides a **minimal, working FastAPI + WebSocket server** for streaming inference  
- Showcases running **Gemma 3 270M** in a **real-time loop** on CPU  
- Serves as a **reference architecture** for larger projects that want to experiment with:  
  - Edge inference (Jetson, Raspberry Pi, etc.)  
  - Realtime agents (voice, chat, embedded systems)  
  - LLM-backed multiplayer / multi-agent systems  

If you’ve ever wanted to see how **incremental LLM state** can be managed outside of “batch mode,” this project is a solid starting point.

---

## Features

- **Hot KV Cache** — Maintains conversation state for fast response times  
- **Live Streaming** — Real-time token streaming via WebSockets  
- **Modular Architecture** — Separate server, input client, and output client  
- **Gemma 3 270M Support** — Small, efficient model for CPU inference  
- **WebSocket API** — Clean separation between input and output  

---

## Environment Setup

### 🤗 Huggingface token

You will need a 🤗 Hugging Face token that allows read access to gemma:270m before proceeding.

* Log into [huggingface](https://huggingface.co) with your account
* Access [gemma:270m](https://huggingface.co/google/gemma-3-270m) and accept any agreements
* Click on your profile | Access Tokens | Create new token
* Create a fine-grained access token with "Read access to contents of all public gated repos you can access" or any other permission you choose
* Copy the generated token and save it as a `.env` file with the following contents

```bash
HUGGINGFACE_TOKEN=hf_mytokenhere
```

You are now ready to use VS Code to develop with this repo.

### Dev Container (Recommended)

We use **VS Code Dev Containers** to ensure a consistent, isolated development environment.  

1. **Install prerequisites**
   - Install [Docker](https://docs.docker.com/get-docker/)
   - Install [Visual Studio Code](https://code.visualstudio.com/)
   - Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
 Installation guide: [VS Code Dev Containers Guide](https://code.visualstudio.com/docs/devcontainers/tutorial)

2. **Open in Dev Container**
   - Open this project folder in VS Code  
   - Press `F1` (or `Ctrl/Cmd+Shift+P`) and run **Dev Containers: Reopen in Container**  
   - VS Code will detect the `.devcontainer/` config, build the container, and reopen inside it  

3. **Rebuild if Needed**
   - If you make changes to `.devcontainer/devcontainer.json` or Docker setup, press `Ctrl/Cmd+Shift+P` → **Dev Containers: Rebuild Container**

### Debugging and running

Use the VS Code Ruyn and Debug panel to launch the compound configuration `Live LLM (web)`. This will launch:

1. **The Live-LLM server** - This may take some time to start the first time as it may need to download the models we use. When ready, you will see 

`INFO:     Application startup complete.`

2. **The web client** - This will launch in a new browser window. Once the server is running, you can refresh this window to connect to the server.

At this point you can 
* Create breakpoints
* Modify the server (it will auto-reload)
* Build the client (Ctrl/Cmd+Shift+B) and refresh it (Shift+F5) to reload changes, etc.

