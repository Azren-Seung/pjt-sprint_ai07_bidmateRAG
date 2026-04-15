"use client";

import { useStore } from "@/store/useStore";
import { TabBar } from "./TabBar";
import { ChatTab } from "./ChatTab";
import { DocumentsTab } from "./DocumentsTab";
import { useCmdBShortcut } from "@/lib/keyboard";
import { Button } from "@/components/ui/button";

export function Sidebar() {
  const collapsed = useStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useStore((s) => s.toggleSidebar);
  const activeTab = useStore((s) => s.activeTab);
  useCmdBShortcut();

  if (collapsed) {
    return (
      <aside className="w-12 border-r border-border">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="mt-2 w-full"
          onClick={toggleSidebar}
          title="사이드바 펼치기 (⌘B)"
        >
          ▶
        </Button>
      </aside>
    );
  }

  return (
    <aside className="flex w-80 flex-col border-r border-border">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <div className="text-sm font-bold">📄 BidMate</div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          title="사이드바 접기 (⌘B)"
        >
          ◀
        </Button>
      </div>
      <TabBar />
      <div className="flex-1 overflow-hidden">
        {activeTab === "chat" ? <ChatTab /> : <DocumentsTab />}
      </div>
    </aside>
  );
}
