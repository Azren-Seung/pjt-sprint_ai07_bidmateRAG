"use client";

import type { DocumentSummary } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { useStore } from "@/store/useStore";
import { cn } from "@/lib/utils";

interface Props {
  doc: DocumentSummary;
}

export function DocumentCard({ doc }: Props) {
  const pinDoc = useStore((s) => s.pinDoc);
  const unpinDoc = useStore((s) => s.unpinDoc);
  const pinnedDocs = useStore((s) => s.pinnedDocs);
  const isPinned = pinnedDocs.some((d) => d.id === doc.id);

  return (
    <button
      type="button"
      onClick={() => (isPinned ? unpinDoc(doc.id) : pinDoc(doc))}
      title={isPinned ? "클릭하여 해제" : "클릭하여 멘션"}
      className={cn(
        "w-full rounded-md border border-border bg-card p-3 text-left transition-colors hover:bg-accent",
        isPinned && "border-primary bg-accent"
      )}
    >
      <div className="truncate text-sm font-medium">{doc.title}</div>
      <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
        <span className="truncate">{doc.agency}</span>
        <span>·</span>
        <span>{doc.budget_label}</span>
      </div>
      <div className="mt-2 flex gap-1">
        {doc.domain && (
          <Badge variant="secondary" className="text-[10px]">
            {doc.domain}
          </Badge>
        )}
        {doc.agency_type && (
          <Badge variant="outline" className="text-[10px]">
            {doc.agency_type}
          </Badge>
        )}
      </div>
    </button>
  );
}
