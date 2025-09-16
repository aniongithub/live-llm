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
    type: 'token' | 'user_input' | 'system';
    data: string;
}

class LiveLLMClient {
    private inputSocket: WebSocket | null = null;
    private outputSocket: WebSocket | null = null;
    private isConnected = false;
    private isConnecting = false;
    private messages: Message[] = [];
    private currentAIMessage = '';
    private maxRetries = 10;
    private retryCount = 0;
    private retryTimeout: number | null = null;

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
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.sendMessage();
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

    private async connect(): Promise<void> {
        if (this.isConnecting || this.isConnected) return;

        this.isConnecting = true;
        this.updateStatus('connecting', 'Connecting...');
        this.retryCount++;

        try {
            const host = window.location.hostname;
            const port = process.env.SERVER_PORT || '8000';
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            
            // Connect to both input and output endpoints
            const inputPromise = this.connectSocket(`${protocol}//${host}:${port}/ws/input`, 'input');
            const outputPromise = this.connectSocket(`${protocol}//${host}:${port}/ws/output`, 'output');

            await Promise.all([inputPromise, outputPromise]);

            this.isConnected = true;
            this.isConnecting = false;
            this.retryCount = 0;
            this.updateStatus('connected', 'Connected');
            this.updateUI();

            console.log('✓ Connected to Live LLM server');

        } catch (error) {
            this.isConnecting = false;
            console.error('Connection failed:', error);
            
            if (this.retryCount < this.maxRetries) {
                const delay = Math.min(Math.pow(2, this.retryCount) * 1000, 10000);
                this.updateStatus('connecting', `Retrying in ${Math.ceil(delay / 1000)}s...`);
                
                this.retryTimeout = window.setTimeout(() => {
                    this.connect();
                }, delay);
            } else {
                this.updateStatus('disconnected', 'Connection failed');
                this.addSystemMessage('Failed to connect to server. Please check if the server is running.');
            }
        }
    }

    private connectSocket(url: string, type: 'input' | 'output'): Promise<void> {
        return new Promise((resolve, reject) => {
            const socket = new WebSocket(url);

            socket.onopen = () => {
                if (type === 'input') {
                    this.inputSocket = socket;
                } else {
                    this.outputSocket = socket;
                }
                resolve();
            };

            socket.onerror = (error) => {
                reject(error);
            };

            socket.onclose = () => {
                if (type === 'input') {
                    this.inputSocket = null;
                } else {
                    this.outputSocket = null;
                }
                
                if (this.isConnected) {
                    this.isConnected = false;
                    this.updateStatus('disconnected', 'Disconnected');
                    this.updateUI();
                    this.addSystemMessage('Connection lost. Attempting to reconnect...');
                    setTimeout(() => this.connect(), 2000);
                }
            };

            if (type === 'output') {
                socket.onmessage = (event) => {
                    this.handleOutputMessage(event.data);
                };
            }
        });
    }

    private handleOutputMessage(data: string): void {
        try {
            const message: ServerMessage = JSON.parse(data);
            
            switch (message.type) {
                case 'user_input':
                    // This is an echo of our own input, we can ignore it since we already display it
                    break;
                    
                case 'token':
                    this.handleAIToken(message.data);
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
        if (this.currentAIMessage === '') {
            // Start of new AI response
            this.startAIMessage();
        }
        
        this.currentAIMessage += token;
        this.updateCurrentAIMessage();
    }

    private startAIMessage(): void {
        const message: Message = {
            type: 'ai',
            content: '',
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
        if (!content || !this.isConnected || !this.inputSocket) return;

        // Add user message to display
        this.addUserMessage(content);

        // Send to server
        try {
            this.inputSocket.send(JSON.stringify({
                type: 'message',
                data: content
            }));

            // Reset current AI message for next response
            this.currentAIMessage = '';

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
        if (!this.isConnected || !this.inputSocket) return;

        try {
            this.inputSocket.send(JSON.stringify({
                type: 'reset'
            }));

            // Clear local messages
            this.messages = [];
            this.currentAIMessage = '';
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
        const isEnabled = this.isConnected;
        
        this.messageInput.disabled = !isEnabled;
        this.sendBtn.disabled = !isEnabled;
        this.resetBtn.disabled = !isEnabled;

        if (isEnabled) {
            this.messageInput.placeholder = 'Type your message here... (Press Ctrl+Enter to send)';
            this.messageInput.focus();
        } else {
            this.messageInput.placeholder = 'Connecting to server...';
        }
    }

    private disconnect(): void {
        if (this.retryTimeout) {
            clearTimeout(this.retryTimeout);
            this.retryTimeout = null;
        }

        if (this.inputSocket) {
            this.inputSocket.close();
            this.inputSocket = null;
        }

        if (this.outputSocket) {
            this.outputSocket.close();
            this.outputSocket = null;
        }

        this.isConnected = false;
        this.isConnecting = false;
    }
}

// Initialize the client when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new LiveLLMClient();
});