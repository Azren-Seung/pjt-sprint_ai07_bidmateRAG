"use client";

import { useState, useRef, KeyboardEvent } from "react";
import { useStore } from "@/store/useStore";
import { Button } from "@/components/ui/button";
import { ChipTray } from "./ChipTray";
import { QuickCommandBar } from "./QuickCommandBar";

export function InputBar() {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const pinnedDocs = useStore((s) => s.pinnedDocs);
  const activeCommand = useStore((s) => s.activeCommand);
  const isLoading = useStore((s) => s.isLoading);
  const sendMessage = useStore((s) => s.sendMessage);

  const requiresDoc = activeCommand?.requires_doc && pinnedDocs.length === 0;
  const requiresMulti =
    activeCommand?.requires_multi_doc && pinnedDocs.length < 2;
  const sendDisabled =
    isLoading ||
    !text.trim() ||
    Boolean(requiresDoc) ||
    Boolean(requiresMulti);

  const disabledReason = requiresMulti
    ? `${activeCommand?.label}는 2개 이상 문서 멘션이 필요합니다`
    : requiresDoc
    ? `${activeCommand?.label}는 문서 멘션이 필요합니다`
    : undefined;

  const handleSend = () => {
    if (sendDisabled) return;
    sendMessage(text);
    setText("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-border bg-background py-2">
      <ChipTray />
      <QuickCommandBar />
      <div className="flex gap-2 px-4">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="질문을 입력하세요. @로 문서 멘션, /로 커맨드 호출"
          className="min-h-[60px] flex-1 resize-none rounded-md border border-border bg-background px-3 py-2 text-sm"
          disabled={isLoading}
        />
        <Button
          type="button"
          onClick={handleSend}
          disabled={sendDisabled}
          title={disabledReason}
        >
          전송
        </Button>
      </div>
    </div>
  );
}
