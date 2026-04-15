"use client";

import { useStore } from "@/store/useStore";
import { MessageList } from "./MessageList";
import { EvidencePanel } from "./EvidencePanel";

export function ChatLayout() {
  const messages = useStore((s) => s.messages);
  const latestAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const citations = latestAssistant?.citations ?? [];

  return (
    <div className="grid h-full grid-cols-[2fr_1fr] gap-6 overflow-y-auto px-6 py-4">
      <div>
        <MessageList />
      </div>
      <div>
        <EvidencePanel citations={citations} />
      </div>
    </div>
  );
}
