import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  isStreaming?: boolean;
  artifacts?: Array<{
    name: string;
    label: string;
    mime_type: string;
    data: string;
  }>;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesWrapperRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    setError(null);

    const userMessage: Message = {
      id: Date.now(),
      text: input,
      isUser: true,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Add initial empty AI message for streaming
    const aiMessageId = Date.now() + 1;
    setMessages((prev) => [
      ...prev,
      { id: aiMessageId, text: '', isUser: false, isStreaming: true },
    ]);

    try {
      const response = await fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, session_id: sessionId }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        const detail = payload?.detail ?? 'Request failed.';
        throw new Error(detail);
      }

      // Handle streaming response (SSE over fetch)
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = '';
      let buffer = '';

      if (reader) {
        let streamDone = false;
        while (!streamDone) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const events = buffer.split(/\n\n/);
          buffer = events.pop() ?? '';

          for (const rawEvent of events) {
            const lines = rawEvent.split(/\r?\n/);
            let dataLines: string[] = [];

            for (const line of lines) {
              if (line.startsWith('data:')) {
                dataLines.push(line.slice(5).trimStart());
              }
            }

            if (dataLines.length === 0) continue;

            const data = dataLines.join('\n');
            if (data === '[DONE]') {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiMessageId
                    ? { ...msg, isStreaming: false }
                    : msg
                )
              );
              streamDone = true;
              break;
            }

            try {
              const parsed = JSON.parse(data);
              if (parsed.type === 'tool_result') {
                const summary = parsed.summary ?? '';
                const artifacts = Array.isArray(parsed.artifacts) ? parsed.artifacts : [];
                setMessages((prev) => [
                  ...prev.filter((msg) => msg.id !== aiMessageId),
                  {
                    id: Date.now(),
                    text: summary,
                    isUser: false,
                    artifacts,
                  },
                ]);
              } else {
                const chunk = parsed.chunk ?? parsed.text ?? parsed.message ?? '';
                if (chunk) {
                  accumulatedText += chunk;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === aiMessageId
                        ? { ...msg, text: accumulatedText }
                        : msg
                    )
                  );
                }
              }
              if (parsed.session_id && parsed.session_id !== sessionId) {
                setSessionId(parsed.session_id);
              }
            } catch {
              // Keep calm and carry on; incomplete data will be retried on next chunk
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      const message =
        error instanceof Error ? error.message : 'Unexpected error.';
      setError(message);
      // Remove the streaming message on error
      setMessages((prev) => prev.filter((msg) => msg.id !== aiMessageId));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const getSenderLabel = (isUser: boolean) => {
    return isUser ? 'You' : 'Trading Copilot';
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Trading Copilot</h1>
        <p className="disclaimer">
          Educational use only. Not financial advice.
        </p>
      </div>

      <div className="messages-wrapper" ref={messagesWrapperRef}>
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">
              <p>Ask about markets, setups, or risk concepts.</p>
            </div>
          )}
          
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`message ${msg.isUser ? 'user' : 'ai'} ${
                msg.isStreaming ? 'streaming' : ''
              }`}
            >
              <div className="message-sender">{getSenderLabel(msg.isUser)}</div>
              <div className="message-bubble">
                {msg.isUser ? (
                  msg.text
                ) : (
                  <>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.text}
                    </ReactMarkdown>
                    {msg.artifacts && msg.artifacts.length > 0 && (
                      <div className="message-artifacts">
                        {msg.artifacts.map((artifact) => (
                          <div key={artifact.name} className="artifact-card">
                            <div className="artifact-label">{artifact.label}</div>
                            <img
                              src={`data:${artifact.mime_type};base64,${artifact.data}`}
                              alt={artifact.label}
                              loading="lazy"
                            />
                          </div>
                        ))}
                      </div>
                    )}
                    {msg.isStreaming && <span className="streaming-cursor" />}
                  </>
                )}
              </div>
            </div>
          ))}
          
          {isLoading && messages.length > 0 && !messages[messages.length - 1]?.isStreaming && (
            <div className="message ai loading">
              <div className="message-sender">Trading Copilot</div>
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          
          {error && (
            <div className="message error">
              <div className="message-sender">Error</div>
              <div className="message-bubble">{error}</div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="input-container-wrapper">
        <div className="input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            disabled={isLoading}
          />
          <button onClick={sendMessage} disabled={isLoading || !input.trim()}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
