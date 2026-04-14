"use client";

import { useStore } from "@/store/useStore";
import { cn } from "@/lib/utils";

export function TabBar() {
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);

  return (
    <div className="flex border-b border-border">
      <button
        type="button"
        onClick={() => setActiveTab("chat")}
        className={cn(
          "flex-1 px-4 py-2 text-sm font-medium transition-colors",
          activeTab === "chat"
            ? "border-b-2 border-primary text-foreground"
            : "text-muted-foreground hover:text-foreground"
        )}
      >
        💬 채팅
      </button>
      <button
        type="button"
        onClick={() => setActiveTab("documents")}
        className={cn(
          "flex-1 px-4 py-2 text-sm font-medium transition-colors",
          activeTab === "documents"
            ? "border-b-2 border-primary text-foreground"
            : "text-muted-foreground hover:text-foreground"
        )}
      >
        📁 보관 문서
      </button>
    </div>
  );
}
