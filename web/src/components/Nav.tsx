"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ThemeToggle from "./ThemeToggle";

const LINKS = [
  { href: "/", label: "สถานะระบบ" },
  { href: "/demo", label: "บัญชีเดโม" },
  { href: "/portfolio", label: "พอร์ต" },
  { href: "/research", label: "ผลทดสอบ" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3 sm:px-6">
        <Link href="/" className="font-semibold text-ink">
          BTC / Gold Trend
        </Link>
        <nav className="flex gap-1 text-sm">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={`rounded-md px-3 py-1.5 ${active ? "bg-surface-2 font-medium text-ink" : "text-ink-2 hover:bg-surface-2"}`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
