import type { Citation } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

export function CitationCard({ citation }: { citation: Citation }) {
  return (
    <div
      id={`cite-${citation.id}`}
      className="rounded-md border border-border bg-card p-3 text-xs"
    >
      <div className="mb-1 flex items-center gap-2">
        <Badge variant="default" className="text-[10px]">
          [{citation.id}]
        </Badge>
        <span className="truncate font-medium">{citation.doc_title}</span>
      </div>
      {citation.section && (
        <div className="mb-2 text-[10px] text-muted-foreground">
          {citation.section}
        </div>
      )}
      <p className="whitespace-pre-wrap text-muted-foreground line-clamp-6">
        {citation.text}
      </p>
    </div>
  );
}
