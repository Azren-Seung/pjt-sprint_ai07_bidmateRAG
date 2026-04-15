"use client";

import { useEffect } from "react";
import { X, ExternalLink } from "lucide-react";
import { useStore } from "@/store/useStore";

export function DocumentPreviewModal() {
  const previewDocId = useStore((s) => s.previewDocId);
  const closePreview = useStore((s) => s.closePreview);
  const doc = useStore((s) =>
    s.previewDocId ? s.documents.find((d) => d.id === s.previewDocId) : undefined
  );

  useEffect(() => {
    if (!previewDocId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closePreview();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [previewDocId, closePreview]);

  if (!previewDocId) return null;

  const pdfUrl = `/api/documents/${encodeURIComponent(previewDocId)}/pdf`;

  return (
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
      onClick={closePreview}
      role="dialog"
      aria-modal="true"
      aria-label="문서 미리보기"
    >
      <div
        className="flex h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-border bg-background shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3 border-b border-border bg-[var(--imessage-chrome)]/90 px-5 py-3 backdrop-blur-md">
          <div className="min-w-0 flex-1">
            <div className="truncate text-[15px] font-semibold">
              {doc?.title ?? "문서 미리보기"}
            </div>
            <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground">
              {doc && (
                <>
                  <span className="truncate">{doc.agency}</span>
                  <span>·</span>
                  <span>{doc.budget_label}</span>
                  {doc.domain && (
                    <>
                      <span>·</span>
                      <span>{doc.domain}</span>
                    </>
                  )}
                  <span>·</span>
                </>
              )}
              <span className="whitespace-nowrap opacity-70">
                💡 PDF 영역 클릭 후 <kbd className="rounded border border-border bg-muted px-1 py-px text-[10px] font-mono">⌘F</kbd> / <kbd className="rounded border border-border bg-muted px-1 py-px text-[10px] font-mono">Ctrl+F</kbd>로 본문 검색
              </span>
            </div>
          </div>
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            title="새 탭에서 열기"
            aria-label="새 탭에서 열기"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <ExternalLink className="size-4" />
          </a>
          <button
            type="button"
            onClick={closePreview}
            aria-label="닫기"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <X className="size-4" />
          </button>
        </div>
        <iframe
          src={pdfUrl}
          title={doc?.title ?? "문서 미리보기"}
          className="flex-1 w-full border-0 bg-muted"
        />
      </div>
    </div>
  );
}
