document.addEventListener('DOMContentLoaded', function () {
    const chatContainer = document.getElementById('chat-interface');
    // Only init chat if student_id is present and chat interface exists
    // (We rely on global studentId variable injected in template)
    if (chatContainer && typeof studentId !== 'undefined' && studentId) {
        initChat();
    }
});

function initChat() {
    addMessage('bot', 'AI 튜터가 수강 이력을 분석 중입니다... ⏳');

    fetch('/api/chat/start/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken()
        },
        body: JSON.stringify({ student_id: studentId })
    })
        .then(response => response.json())
        .then(data => {
            // Clear loading message
            const chatWindow = document.getElementById('chat-window');
            chatWindow.innerHTML = '';

            if (data.error) {
                addMessage('bot', '오류가 발생했습니다: ' + data.error);
            } else {
                addMessage('bot', data.message);
                renderChoices(data.choices);
            }
        })
        .catch(err => {
            addMessage('bot', '서버 통신 오류가 발생했습니다.');
            console.error(err);
        });
}

function handleChoice(value, label) {
    // 1. Display user choice as message
    addMessage('user', label);

    // 2. Remove choice buttons
    const choicesDiv = document.querySelector('.choices-container:last-child');
    if (choicesDiv) choicesDiv.remove();

    // 3. Check for client-side actions
    if (value === 'RESET') {
        // Special case: Refresh page or Show Table
        // Here we just refresh to show table if that's the logic, or show hidden table
        document.getElementById('base-results-container').style.display = 'block';
        addMessage('bot', '전체 추천 결과를 아래에 표시했습니다! 👇');
        document.getElementById('base-results-container').scrollIntoView({ behavior: 'smooth' });
        return;
    }

    if (label.includes('결과 확인') || value === 'show_table') {
        // Show results
        // In real scenario, we might want to fetch *filtered* table from server.
        // But for now, let's just show the table we have (or reload page with filters).
        // Since we used AJAX for chat, the table HTML isn't updated.
        // OPTION: Reload page with parameters? Or fetch HTML?
        // Simplest: Submit the hidden form with filters?
        // Actually, the ChatService `filtered` IDs are in session.
        // We can redirect to a URL that renders table from session state?

        // Let's implement a simple "Show Table" via reload or revealing.
        // Since this is the final step:
        addMessage('bot', '잠시만요, 결과를 정리해서 보여드릴게요... 📊');
        setTimeout(() => {
            // Submit a form to refresh page with 'chat_result' mode?
            // Or simpler: The user asked to show table.
            // We can trigger the existing filtering form submission or just reveal.
            // But existing table contains "Top 60". We want "Filtered Result".

            // Workaround: Reload page? Or maybe we simply reveal the existing table 
            // IF the chat didn't change filters (Top 80 case).
            // But if we filtered, we need new data.

            // Best approach: Submit the form to `/recommend/` with the current filters?
            // But filters are in session.
            // Let's reload page and have view render based on session?
            // Or just redirect to home?

            window.location.href = window.location.pathname + "?show_chat_result=true";
        }, 1000);
        return;
    }

    // 4. Send to API
    addMessage('bot', '입력 확인 중... 💬'); // Typing indicator replacement

    fetch('/api/chat/message/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken()
        },
        body: JSON.stringify({ value: value })
    })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator (last bot message)
            const chatWindow = document.getElementById('chat-window');
            chatWindow.lastElementChild.remove();

            if (data.error) {
                addMessage('bot', '오류: ' + data.error);
            } else {
                addMessage('bot', data.message);
                if (data.choices) renderChoices(data.choices);

                if (data.action === 'show_table') {
                    // Determine logic to show table
                    // Maybe the choices button triggers it? 
                    // Wait, if action is show_table, we typically present a button to do it.
                }
            }
        });
}

function addMessage(sender, text) {
    const chatWindow = document.getElementById('chat-window');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;
    // Allow basic formatting for bot
    msgDiv.innerHTML = text.replace(/\n/g, '<br>');
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function renderChoices(choices) {
    if (!choices || choices.length === 0) return;

    const chatWindow = document.getElementById('chat-window');
    const container = document.createElement('div');
    container.className = 'choices-container';

    choices.forEach(choice => {
        const btn = document.createElement('button');
        btn.className = 'chat-btn';
        btn.textContent = choice.label;
        btn.onclick = () => handleChoice(choice.value, choice.label);
        container.appendChild(btn);
    });

    chatWindow.appendChild(container);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function getCsrfToken() {
    return document.querySelector('[name=csrfmiddlewaretoken]').value;
}
