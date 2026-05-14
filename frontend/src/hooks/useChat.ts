import { useCallback, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { APIError, clearConversation, sendChat, streamChat, type SourceChunk } from "../lib/api";

export interface ChatMessage {
  id: string; role: "user" | "assistant"; content: string;
  sources?: SourceChunk[]; latency_ms?: number; isStreaming?: boolean; error?: boolean;
}

export interface ChatSettings {
  useRAG: boolean; stream: boolean; provider: "openai" | "huggingface";
  model: string; temperature: number; maxTokens: number; systemPrompt: string;
}

const DEFAULT_SETTINGS: ChatSettings = {
  useRAG: true, stream: true, provider: "openai",
  model: "", temperature: 0.7, maxTokens: 1024, systemPrompt: "",
};

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [conversationId] = useState(() => uuidv4());
  const abortRef = useRef<boolean>(false);

  const appendMessage = (msg: ChatMessage) => setMessages(prev => [...prev, msg]);

  const updateLastAssistant = (update: Partial<ChatMessage>) =>
    setMessages(prev => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") { next[i] = { ...next[i], ...update }; break; }
      }
      return next;
    });

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;
    abortRef.current = false;
    appendMessage({ id: uuidv4(), role: "user", content: text.trim() });
    setIsLoading(true);

    const req = {
      message: text.trim(), conversation_id: conversationId,
      use_rag: settings.useRAG, provider: settings.provider,
      model: settings.model || undefined,
      system_prompt: settings.systemPrompt || undefined,
      temperature: settings.temperature, max_tokens: settings.maxTokens,
    };

    if (settings.stream) {
      const assistantId = uuidv4();
      appendMessage({ id: assistantId, role: "assistant", content: "", isStreaming: true });
      await streamChat(req,
        (token) => {
          if (abortRef.current) return;
          setMessages(prev => {
            const next = [...prev];
            const idx = next.findIndex(m => m.id === assistantId);
            if (idx !== -1) next[idx] = { ...next[idx], content: next[idx].content + token };
            return next;
          });
        },
        () => { updateLastAssistant({ isStreaming: false }); setIsLoading(false); },
        (err) => { updateLastAssistant({ content: `Error: ${err.message}`, isStreaming: false, error: true }); setIsLoading(false); }
      );
    } else {
      try {
        const resp = await sendChat(req);
        appendMessage({ id: uuidv4(), role: "assistant", content: resp.message.content, sources: resp.sources, latency_ms: resp.latency_ms });
      } catch (err) {
        appendMessage({ id: uuidv4(), role: "assistant", content: `Error: ${err instanceof APIError ? err.detail : "Unexpected error"}`, error: true });
      } finally { setIsLoading(false); }
    }
  }, [conversationId, isLoading, settings]);

  const clearChat = useCallback(async () => { await clearConversation(conversationId); setMessages([]); }, [conversationId]);
  const stopStreaming = useCallback(() => { abortRef.current = true; setIsLoading(false); updateLastAssistant({ isStreaming: false }); }, []);

  return { messages, isLoading, settings, setSettings, sendMessage, clearChat, stopStreaming, conversationId };
}
