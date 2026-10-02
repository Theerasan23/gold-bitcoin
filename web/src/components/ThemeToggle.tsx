"use client";

import { useSyncExternalStore } from "react";

type Mode = "system" | "light" | "dark";
const LABEL: Record<Mode, string> = { system: "ตามระบบ", light: "สว่าง", dark: "มืด" };
const NEXT: Record<Mode, Mode> = { system: "light", light: "dark", dark: "system" };

// ธีมเก็บที่ <html data-theme> (ตั้งก่อนหน้าเว็บแสดงใน layout) — อ่าน/ฟังจาก DOM โดยตรง
function subscribe(cb: () => void) {
  const mo = new MutationObserver(cb);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => mo.disconnect();
}
function getMode(): Mode {
  const t = document.documentElement.dataset.theme;
  return t === "light" || t === "dark" ? t : "system";
}

export default function ThemeToggle() {
  const mode = useSyncExternalStore(subscribe, getMode, () => "system" as Mode);

  function cycle() {
    const next = NEXT[mode];
    if (next === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = next;
    try {
      if (next === "system") localStorage.removeItem("theme");
      else localStorage.setItem("theme", next);
    } catch {}
  }

  return (
    <button
      type="button"
      onClick={cycle}
      className="rounded-md border border-line px-3 py-1.5 text-sm text-ink-2 hover:bg-surface-2"
      aria-label={`ธีม: ${LABEL[mode]} (กดเพื่อเปลี่ยน)`}
    >
      ธีม: {LABEL[mode]}
    </button>
  );
}
