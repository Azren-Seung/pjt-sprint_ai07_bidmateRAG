"use client";

import { useStore } from "@/store/useStore";
import { UserMessage } from "./UserMessage";
import { AssistantMessage } from "./AssistantMessage";

export function MessageList() {
  const messages = useStore((s) => s.messages);
  const isLoading = useStore((s) => s.isLoading);

  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        <div className="text-center">
          <div className="text-2xl">📄</div>
          <div className="mt-2">RFP 질의를 시작하세요.</div>
          <div className="mt-1 text-xs">
            사이드바에서 문서를 클릭하거나 입력창에 <code>@</code> / <code>/</code>를 입력하세요.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {messages.map((m) =>
        m.role === "user" ? (
          <UserMessage key={m.id} message={m} />
        ) : (
          <AssistantMessage key={m.id} message={m} />
        )
      )}
      {isLoading && (
        <div className="rounded-md bg-muted px-4 py-3 text-sm text-muted-foreground">
          답변 생성 중... (5~15초)
        </div>
      )}
    </div>
  );
}
