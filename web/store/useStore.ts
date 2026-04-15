import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type {
  DocumentSummary,
  SlashCommandMeta,
  Message,
  QueryResponse,
} from "@/lib/types";
import { postQuery } from "@/lib/api";

interface DocumentFilters {
  domain?: string;
  agencyType?: string;
  budgetRange?: string;
}

interface Store {
  sidebarCollapsed: boolean;
  activeTab: "chat" | "documents";

  documents: DocumentSummary[];
  documentSearchQuery: string;
  documentFilters: DocumentFilters;

  pinnedDocs: DocumentSummary[];
  activeCommand: SlashCommandMeta | null;

  messages: Message[];
  isLoading: boolean;
  lastError: string | null;

  providerConfig: string;
  chunkingConfig: string | null;
  topK: number;

  setSidebarCollapsed: (v: boolean) => void;
  toggleSidebar: () => void;
  setActiveTab: (tab: "chat" | "documents") => void;

  setDocuments: (docs: DocumentSummary[]) => void;
  setDocumentSearchQuery: (q: string) => void;
  setDocumentFilters: (f: DocumentFilters) => void;

  pinDoc: (doc: DocumentSummary) => void;
  unpinDoc: (docId: string) => void;
  setCommand: (cmd: SlashCommandMeta | null) => void;
  clearContext: () => void;

  newChat: () => void;
  sendMessage: (text: string) => Promise<void>;
}

export const useStore = create<Store>()(
  persist(
    (set, get) => ({
      sidebarCollapsed: false,
      activeTab: "documents",

      documents: [],
      documentSearchQuery: "",
      documentFilters: {},

      pinnedDocs: [],
      activeCommand: null,

      messages: [],
      isLoading: false,
      lastError: null,

      providerConfig: "openai_gpt5mini",
      chunkingConfig: null,
      topK: 5,

      setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),
      toggleSidebar: () =>
        set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setActiveTab: (tab) => set({ activeTab: tab }),

      setDocuments: (docs) => set({ documents: docs }),
      setDocumentSearchQuery: (q) => set({ documentSearchQuery: q }),
      setDocumentFilters: (f) => set({ documentFilters: f }),

      pinDoc: (doc) => {
        const existing = get().pinnedDocs;
        if (existing.some((d) => d.id === doc.id)) return;
        set({ pinnedDocs: [...existing, doc] });
      },
      unpinDoc: (docId) =>
        set((s) => ({
          pinnedDocs: s.pinnedDocs.filter((d) => d.id !== docId),
        })),
      setCommand: (cmd) => set({ activeCommand: cmd }),
      clearContext: () => set({ pinnedDocs: [], activeCommand: null }),

      newChat: () =>
        set({
          messages: [],
          lastError: null,
          pinnedDocs: [],
          activeCommand: null,
        }),

      sendMessage: async (text) => {
        const state = get();
        const userMessage: Message = {
          id: `user-${Date.now()}`,
          role: "user",
          content: text,
        };
        set({
          messages: [...state.messages, userMessage],
          isLoading: true,
          lastError: null,
        });

        try {
          const response: QueryResponse = await postQuery({
            question: text,
            provider_config: state.providerConfig,
            chunking_config: state.chunkingConfig,
            mentioned_doc_ids: state.pinnedDocs.map((d) => d.id),
            command: state.activeCommand?.id ?? null,
            top_k: state.topK,
            max_context_chars: 8000,
          });

          const assistantMessage: Message = {
            id: `assistant-${Date.now()}`,
            role: "assistant",
            content: response.answer,
            citations: response.citations,
            metadata: response.metadata,
          };
          set((s) => ({
            messages: [...s.messages, assistantMessage],
            isLoading: false,
          }));

          if (response.metadata.command_applied === "초기화") {
            set({ pinnedDocs: [], activeCommand: null });
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          set((s) => ({
            messages: [
              ...s.messages,
              {
                id: `error-${Date.now()}`,
                role: "assistant",
                content: `오류: ${message}`,
                error: message,
              },
            ],
            isLoading: false,
            lastError: message,
          }));
        }
      },
    }),
    {
      name: "bidmate-session",
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        messages: state.messages,
        pinnedDocs: state.pinnedDocs,
        activeCommand: state.activeCommand,
        sidebarCollapsed: state.sidebarCollapsed,
        activeTab: state.activeTab,
      }),
    }
  )
);
