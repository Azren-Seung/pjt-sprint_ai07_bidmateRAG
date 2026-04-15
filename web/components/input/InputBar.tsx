"use client";

import { useState } from "react";
import { useStore } from "@/store/useStore";
import { Button } from "@/components/ui/button";
import { ChipTray } from "./ChipTray";
import { QuickCommandBar } from "./QuickCommandBar";
import { MentionTextarea } from "./MentionTextarea";

function stripMentionMarkup(raw: string): string {
  // react-mentions stores markup like `@[display](id)` or `/[display](id)`.
  // On submit we strip all mention/command tokens since they are captured
  // separately in the Zustand store as pinnedDocs / activeCommand.
  return raw
    .replace(/@\[[^\]]+\]\([^)]+\)/g, "")
    .replace(/\/\[[^\]]+\]\([^)]+\)/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function InputBar() {
  const [text, setText] = useState("");

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
    const cleaned = stripMentionMarkup(text);
    if (!cleaned) return;
    sendMessage(cleaned);
    setText("");
  };

  return (
    <div className="border-t border-border bg-background py-2">
      <ChipTray />
      <QuickCommandBar />
      <div className="flex gap-2 px-4">
        <div className="flex-1">
          <MentionTextarea
            value={text}
            onChange={setText}
            onEnter={handleSend}
            disabled={isLoading}
          />
        </div>
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
