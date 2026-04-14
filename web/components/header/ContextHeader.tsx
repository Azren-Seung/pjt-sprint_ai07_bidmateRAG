"use client";

import { useStore } from "@/store/useStore";
import { QuickFactsCard } from "./QuickFactsCard";
import { Badge } from "@/components/ui/badge";

export function ContextHeader() {
  const pinnedDocs = useStore((s) => s.pinnedDocs);
  const activeCommand = useStore((s) => s.activeCommand);

  if (pinnedDocs.length === 0 && !activeCommand) return null;

  return (
    <div className="border-b border-border bg-background px-6 py-3">
      {pinnedDocs.length === 1 && <QuickFactsCard docId={pinnedDocs[0].id} />}
      {pinnedDocs.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {pinnedDocs.map((d) => (
            <Badge key={d.id} variant="secondary" className="text-xs">
              📄 {d.title}
            </Badge>
          ))}
        </div>
      )}
      {activeCommand && (
        <div className="mt-2 text-xs text-muted-foreground">
          활성 커맨드: <Badge variant="default">{activeCommand.label}</Badge>
        </div>
      )}
    </div>
  );
}
