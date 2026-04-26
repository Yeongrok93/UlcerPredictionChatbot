"use client";

import { useEffect, useRef, useState } from "react";

type Role = "user" | "assistant";
type Message = { role: Role; content: string };

const GREETING: Message = {
  role: "assistant",
  content:
    "안녕하세요. 욕창(Pressure Injury) 예측 모델 챗봇입니다.\n모델, 입력 변수, 성능, 사용 방법 등 자료에 있는 내용을 무엇이든 물어보세요.",
};

const TYPE_INTERVAL_MS = 18;
const TYPE_CHARS_PER_TICK = 2;

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([GREETING]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [typingIdx, setTypingIdx] = useState<number | null>(null);
  const [typedChars, setTypedChars] = useState(0);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, typedChars, loading]);

  useEffect(() => {
    if (typingIdx === null) return;
    const target = messages[typingIdx];
    if (!target) {
      setTypingIdx(null);
      return;
    }
    if (typedChars >= target.content.length) {
      setTypingIdx(null);
      return;
    }
    const t = setTimeout(
      () => setTypedChars((c) => Math.min(c + TYPE_CHARS_PER_TICK, target.content.length)),
      TYPE_INTERVAL_MS,
    );
    return () => clearTimeout(t);
  }, [typingIdx, typedChars, messages]);

  async function send() {
    const content = input.trim();
    if (!content || loading) return;
    const next: Message[] = [...messages, { role: "user", content }];
    setMessages(next);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: next }),
      });
      const data = await res.json();
      const reply: string = data.error
        ? `오류가 발생했습니다: ${data.error}`
        : data.reply || "(빈 응답)";
      const after = [...next, { role: "assistant" as Role, content: reply }];
      setMessages(after);
      setTypedChars(0);
      setTypingIdx(after.length - 1);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      const after = [
        ...next,
        { role: "assistant" as Role, content: `네트워크 오류: ${msg}` },
      ];
      setMessages(after);
      setTypedChars(0);
      setTypingIdx(after.length - 1);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <main
      className="flex flex-col bg-slate-50"
      style={{ height: "100dvh" }}
    >
      <header
        className="bg-white border-b shadow-sm px-4 py-3"
        style={{ paddingTop: "calc(env(safe-area-inset-top, 0px) + 0.75rem)" }}
      >
        <h1 className="text-base sm:text-lg font-bold text-slate-800">
          욕창 예측 모델 챗봇
        </h1>
        <p className="text-[11px] sm:text-xs text-slate-500">
          아산병원 RAG · GPT-4o · 자료 기반 질의응답
        </p>
      </header>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-3 sm:px-4 py-4 sm:py-6"
        style={{ overscrollBehavior: "contain", WebkitOverflowScrolling: "touch" } as React.CSSProperties}
      >
        <div className="mx-auto max-w-3xl space-y-3 sm:space-y-4">
          {messages.map((m, i) => {
            const isTyping = i === typingIdx;
            const shown = isTyping ? m.content.slice(0, typedChars) : m.content;
            return (
              <Bubble
                key={i}
                role={m.role}
                content={shown}
                showCaret={isTyping}
              />
            );
          })}
          {loading && <Bubble role="assistant" content="생각 중…" muted />}
        </div>
      </div>

      <div
        className="border-t bg-white px-3 sm:px-4 py-2 sm:py-3"
        style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 0px) + 0.5rem)" }}
      >
        <div className="mx-auto max-w-3xl flex gap-2 items-end">
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="질문을 입력하고 Enter (줄바꿈은 Shift+Enter)"
            disabled={loading}
            className="flex-1 resize-none border border-slate-300 rounded-2xl px-4 py-2.5 text-base focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100 max-h-40 leading-snug"
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:bg-slate-300 text-white text-sm font-medium px-5 py-2.5 rounded-2xl transition shrink-0 min-h-[44px]"
          >
            전송
          </button>
        </div>
      </div>
    </main>
  );
}

function Bubble({
  role,
  content,
  muted,
  showCaret,
}: {
  role: Role;
  content: string;
  muted?: boolean;
  showCaret?: boolean;
}) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={[
          "max-w-[88%] sm:max-w-[85%] px-4 py-2.5 rounded-2xl text-[15px] sm:text-sm leading-relaxed whitespace-pre-wrap break-words shadow-sm",
          isUser
            ? "bg-blue-600 text-white rounded-br-md"
            : "bg-white text-slate-800 border border-slate-200 rounded-bl-md",
          muted ? "italic text-slate-500" : "",
        ].join(" ")}
      >
        {content}
        {showCaret && <span className="typing-caret" aria-hidden="true" />}
      </div>
    </div>
  );
}
