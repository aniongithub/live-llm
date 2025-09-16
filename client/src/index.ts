/**
 * Live LLM Web Client
 * Combines input and output functionality in a single web interface
 */

// Import styles
import './index.css';

interface Message {
    type: 'user' | 'ai' | 'system';
    content: string;
    timestamp: Date;
}

interface ServerMessage {
    type: 'token' | 'user_input' | 'system' | 'end';
    data: string;
}

class LiveLLMClient {
    private socket: WebSocket | null = null;
    private isConnected = false;
    private messages: Message[] = [];
    private currentAIMessage = '';
    private isStreamingAI = false;

    // DOM elements
    private messagesContainer!: HTMLElement;
    private messageInput!: HTMLTextAreaElement;
    private sendBtn!: HTMLButtonElement;
    private resetBtn!: HTMLButtonElement;
    private statusDot!: HTMLElement;
    private statusText!: HTMLElement;
    private charCount!: HTMLElement;

    constructor() {
        this.bindElements();
        this.setupEventListeners();
        this.connect();
    }

    private bindElements(): void {
        this.messagesContainer = document.getElementById('messages')!;
        this.messageInput = document.getElementById('messageInput') as HTMLTextAreaElement;
        this.sendBtn = document.getElementById('sendBtn') as HTMLButtonElement;
        this.resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
        this.statusDot = document.getElementById('statusDot')!;
        this.statusText = document.getElementById('statusText')!;
        this.charCount = document.getElementById('charCount')!;
    }

    private setupEventListeners(): void {
        // Send button click
        this.sendBtn.addEventListener('click', () => this.sendMessage());

        // Enter key handling
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.ctrlKey) {
                e.preventDefault();
                this.sendMessage();
            } else if (e.key === 'Enter' && e.ctrlKey) {
                // Allow default behavior (carriage return)
                // Don't prevent default, let the textarea handle it
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.updateCharCount();
            this.autoResizeTextarea();
        });

        // Reset button
        this.resetBtn.addEventListener('click', () => this.resetConversation());

        // Window beforeunload to cleanup connections
        window.addEventListener('beforeunload', () => this.disconnect());
    }

    private updateCharCount(): void {
        const length = this.messageInput.value.length;
        this.charCount.textContent = `${length}/2000`;
    }

    private autoResizeTextarea(): void {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    private connect(): void {
        if (this.isConnected) return;

        const host = window.location.hostname;
        const port = process.env.SERVER_PORT || '8000';
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        
        this.socket = new WebSocket(`${protocol}//${host}:${port}/ws`);
        
        this.socket.onopen = () => {
            this.isConnected = true;
            this.updateStatus('connected', 'Connected');
            this.updateUI();
            console.log('✓ Connected to Live LLM server');
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            // If we can't connect on load, just reload the page
            if (!this.isConnected) {
                console.log('Failed to connect, reloading page...');
                setTimeout(() => window.location.reload(), 2000);
            }
        };

        this.socket.onclose = () => {
            this.socket = null;
            this.isConnected = false;
            this.updateStatus('disconnected', 'Disconnected');
            this.updateUI();
        };

        this.socket.onmessage = (event) => {
            this.handleMessage(event.data);
        };
    }

    private handleMessage(data: string): void {
        try {
            const message: ServerMessage = JSON.parse(data);
            
            switch (message.type) {
                case 'user_input':
                    // This is an echo of our own input, we can ignore it since we already display it
                    break;
                    
                case 'token':
                    this.handleAIToken(message.data);
                    break;
                    
                case 'end':
                    this.handleStreamEnd();
                    break;
                    
                case 'system':
                    this.addSystemMessage(message.data);
                    break;
            }
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    }

    private handleAIToken(token: string): void {
        console.log('Received token:', JSON.stringify(token), 'isStreamingAI:', this.isStreamingAI);
        
        if (!this.isStreamingAI) {
            console.log('Starting new AI message');
            // Start of new AI response
            this.isStreamingAI = true;
            this.currentAIMessage = token; // Initialize with first token
            this.startAIMessage();
            // Disable input while AI is responding
            this.updateUI();
        } else {
            this.currentAIMessage += token;
            this.updateCurrentAIMessage();
        }
    }

    private handleStreamEnd(): void {
        console.log('Stream ended, isStreamingAI:', this.isStreamingAI);
        if (this.isStreamingAI) {
            // Remove typing indicator from current AI message
            const messageElements = this.messagesContainer.querySelectorAll('.message');
            const lastMessageElement = messageElements[messageElements.length - 1];
            if (lastMessageElement && lastMessageElement.classList.contains('ai-message')) {
                const typingIndicator = lastMessageElement.querySelector('.typing-indicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }
            
            // Reset streaming state
            this.isStreamingAI = false;
            this.currentAIMessage = '';
            
            // Re-enable input
            this.updateUI();
        }
    }

    private startAIMessage(): void {
        console.log('Creating new AI message with content:', JSON.stringify(this.currentAIMessage));
        const message: Message = {
            type: 'ai',
            content: this.currentAIMessage, // Use current content instead of empty string
            timestamp: new Date()
        };
        
        this.messages.push(message);
        this.renderMessage(message, true); // true indicates it's streaming
    }

    private updateCurrentAIMessage(): void {
        const lastMessage = this.messages[this.messages.length - 1];
        if (lastMessage && lastMessage.type === 'ai') {
            lastMessage.content = this.currentAIMessage;
            
            // Update the displayed message
            const messageElements = this.messagesContainer.querySelectorAll('.message');
            const lastMessageElement = messageElements[messageElements.length - 1];
            if (lastMessageElement) {
                const contentElement = lastMessageElement.querySelector('.message-content');
                if (contentElement) {
                    contentElement.textContent = this.currentAIMessage;
                }
            }
            
            this.scrollToBottom();
        }
    }

    private sendMessage(): void {
        const content = this.messageInput.value.trim();
        if (!content || !this.isConnected || !this.socket || this.isStreamingAI) return;

        // Add user message to display
        this.addUserMessage(content);

        // Send to server
        try {
            this.socket.send(JSON.stringify({
                type: 'message',
                data: content
            }));

            // Clear input
            this.messageInput.value = '';
            this.updateCharCount();
            this.autoResizeTextarea();

        } catch (error) {
            console.error('Error sending message:', error);
            this.addSystemMessage('Failed to send message. Please try again.');
        }
    }

    private resetConversation(): void {
        if (!this.isConnected || !this.socket) return;

        try {
            this.socket.send(JSON.stringify({
                type: 'reset'
            }));

            // Clear local messages and reset state
            this.messages = [];
            this.currentAIMessage = '';
            this.isStreamingAI = false;
            this.updateUI();
            this.renderMessages();

        } catch (error) {
            console.error('Error resetting conversation:', error);
            this.addSystemMessage('Failed to reset conversation. Please try again.');
        }
    }

    private addUserMessage(content: string): void {
        const message: Message = {
            type: 'user',
            content,
            timestamp: new Date()
        };
        
        this.messages.push(message);
        this.renderMessage(message);
        this.scrollToBottom();
    }

    private addSystemMessage(content: string): void {
        const message: Message = {
            type: 'system',
            content,
            timestamp: new Date()
        };
        
        this.messages.push(message);
        this.renderMessage(message);
        this.scrollToBottom();
    }

    private renderMessages(): void {
        this.messagesContainer.innerHTML = '';
        
        if (this.messages.length === 0) {
            this.messagesContainer.innerHTML = `
                <div class="welcome-message">
                    <p>Welcome to Live LLM! Start a conversation by typing a message below.</p>
                </div>
            `;
            return;
        }

        this.messages.forEach(message => this.renderMessage(message));
        this.scrollToBottom();
    }

    private renderMessage(message: Message, isStreaming = false): void {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.type}-message`;

        const timeStr = message.timestamp.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        let authorName: string;
        switch (message.type) {
            case 'user':
                authorName = 'You';
                break;
            case 'ai':
                authorName = 'AI';
                break;
            case 'system':
                authorName = 'System';
                break;
        }

        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="message-author">${authorName}</span>
                <span class="message-time">${timeStr}</span>
            </div>
            <div class="message-content">${message.content}</div>
            ${isStreaming ? '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>' : ''}
        `;

        this.messagesContainer.appendChild(messageDiv);
        
        // Remove welcome message if it exists
        const welcomeMessage = this.messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
    }

    private scrollToBottom(): void {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    private updateStatus(status: 'connected' | 'connecting' | 'disconnected', text: string): void {
        this.statusDot.className = `status-dot ${status}`;
        this.statusText.textContent = text;
    }

    private updateUI(): void {
        const isEnabled = this.isConnected && !this.isStreamingAI;
        
        this.messageInput.disabled = !isEnabled;
        this.sendBtn.disabled = !isEnabled;
        this.resetBtn.disabled = !isEnabled;

        if (isEnabled) {
            this.messageInput.placeholder = 'Type your message here... (Press Enter to send, Ctrl+Enter for new line)';
            this.messageInput.focus();
        } else if (this.isStreamingAI) {
            this.messageInput.placeholder = 'AI is responding...';
        } else {
            this.messageInput.placeholder = 'Connecting to server...';
        }
    }

    private disconnect(): void {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
        this.isConnected = false;
    }
}

// Initialize the client when the DOM is loaded
let clientInstance: LiveLLMClient | null = null;

document.addEventListener('DOMContentLoaded', () => {
    if (clientInstance) {
        console.log('Client instance already exists, skipping initialization');
        return;
    }
    
    console.log('Creating LiveLLMClient instance');
    clientInstance = new LiveLLMClient();
});