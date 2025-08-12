document.addEventListener('DOMContentLoaded', function() {
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    const status = document.getElementById('status');
    
    function addMessage(text, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'ai-message');
        messageDiv.textContent = text;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    
    async function sendQuestion() {
        const question = userInput.value.trim();
        if (!question) return;
        
        addMessage(question, true);
        userInput.value = '';
        
        status.textContent = 'Thinking...';
        status.classList.add('typing-indicator');
        
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question })
            });
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            
            const data = await response.json();
            addMessage(data.answer, false);
            
        } catch (error) {
            addMessage("Sorry, I couldn't process your request. Please try again.", false);
            console.error('Error:', error);
        } finally {
            status.textContent = '';
            status.classList.remove('typing-indicator');
        }
    }
    
    function clearChat() {
        chatHistory.innerHTML = '';
        addMessage("Hello! I'm your IT Infrastructure Assistant. Ask me anything about enterprise computing, virtualization, or end-user computing solutions.", false);
    }
    
    // Event listeners
    sendBtn.addEventListener('click', sendQuestion);
    clearBtn.addEventListener('click', clearChat);
    
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });
});