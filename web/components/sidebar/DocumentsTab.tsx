"use client";

import { useEffect, useMemo, useRef } from "react";
import { useStore } from "@/store/useStore";
import { listDocuments } from "@/lib/api";
import { searchDocuments, filterDocuments } from "@/lib/fuse-search";
import { DocumentCard } from "./DocumentCard";
import { DocumentFilters } from "./DocumentFilters";
import { Input } from "@/components/ui/input";
import { useCmdKShortcut } from "@/lib/keyboard";

export function DocumentsTab() {
  const documents = useStore((s) => s.documents);
  const setDocuments = useStore((s) => s.setDocuments);
  const searchQuery = useStore((s) => s.documentSearchQuery);
  const setSearchQuery = useStore((s) => s.setDocumentSearchQuery);
  const filters = useStore((s) => s.documentFilters);

  const searchInputRef = useRef<HTMLInputElement>(null);
  useCmdKShortcut(searchInputRef);

  useEffect(() => {
    if (documents.length === 0) {
      listDocuments()
        .then((data) => setDocuments(data.documents))
        .catch((err) => console.error("failed to load documents", err));
    }
  }, [documents.length, setDocuments]);

  const visible = useMemo(() => {
    const filtered = filterDocuments(documents, filters);
    return searchDocuments(filtered, searchQuery);
  }, [documents, filters, searchQuery]);

  return (
    <div className="flex h-full flex-col">
      <div className="p-4 pb-2">
        <Input
          ref={searchInputRef}
          type="search"
          placeholder="🔍 문서 검색 (⌘K)"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>
      <DocumentFilters documents={documents} />
      <div className="mt-2 flex-1 overflow-y-auto px-4 pb-4">
        <div className="mb-2 text-xs text-muted-foreground">
          {visible.length}건 / 총 {documents.length}건
        </div>
        <div className="flex flex-col gap-2">
          {visible.map((doc) => (
            <DocumentCard key={doc.id} doc={doc} />
          ))}
        </div>
      </div>
    </div>
  );
}
