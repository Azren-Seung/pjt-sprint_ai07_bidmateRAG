"use client";

import { useEffect, RefObject } from "react";
import { useStore } from "@/store/useStore";

/** ⌘K / Ctrl+K 단축키: 사이드바 → 보관 문서 탭 → 검색창 포커스 */
export function useCmdKShortcut(
  searchInputRef: RefObject<HTMLInputElement | null>
): void {
  const setSidebarCollapsed = useStore((s) => s.setSidebarCollapsed);
  const setActiveTab = useStore((s) => s.setActiveTab);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSidebarCollapsed(false);
        setActiveTab("documents");
        // 다음 tick에 포커스 (탭 전환 후)
        setTimeout(() => {
          searchInputRef.current?.focus();
          searchInputRef.current?.select();
        }, 50);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [setSidebarCollapsed, setActiveTab, searchInputRef]);
}

/** ⌘B / Ctrl+B 단축키: 사이드바 토글 */
export function useCmdBShortcut(): void {
  const toggleSidebar = useStore((s) => s.toggleSidebar);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        toggleSidebar();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [toggleSidebar]);
}
