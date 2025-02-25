document.getElementById('chat-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    if (message) {
        appendMessage('user', message);
        input.value = '';
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            appendMessage('bot', data.response);
        } catch (error) {
            appendMessage('bot', 'Error: ' + error.message);
        }
    }
});

function appendMessage(sender, text) {
    const conversation = document.getElementById('conversation');
    const messageElement = document.createElement('div');
    messageElement.classList.add(sender + '-message');
    messageElement.textContent = text;
    conversation.appendChild(messageElement);
    conversation.scrollTop = conversation.scrollHeight;
}