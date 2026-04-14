import type { Message } from "@/lib/types";

export function UserMessage({ message }: { message: Message }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground">
        {message.content}
      </div>
    </div>
  );
}
