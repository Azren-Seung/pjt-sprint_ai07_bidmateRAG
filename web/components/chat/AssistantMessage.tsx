import type { Message } from "@/lib/types";
import { MarkdownWithCitations } from "./MarkdownWithCitations";

export function AssistantMessage({ message }: { message: Message }) {
  if (message.error) {
    return (
      <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-2 text-sm text-destructive">
        {message.content}
      </div>
    );
  }
  return (
    <div className="rounded-lg bg-card px-4 py-3 text-sm">
      <MarkdownWithCitations content={message.content} />
      {message.metadata && (
        <div className="mt-2 flex gap-3 text-[11px] text-muted-foreground">
          <span>🤖 {message.metadata.model}</span>
          <span>⏱ {(message.metadata.latency_ms / 1000).toFixed(1)}s</span>
          <span>🎯 {message.metadata.token_usage.total ?? 0} 토큰</span>
        </div>
      )}
    </div>
  );
}
