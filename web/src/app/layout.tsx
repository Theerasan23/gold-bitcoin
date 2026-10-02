import type { Metadata } from "next";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "BTC / Gold Trend",
  description: "สถานะระบบเทรนด์ BTC และทองคำ ผลพอร์ต และผลทดสอบกลยุทธ์",
};

// ตั้งธีมก่อนหน้าเว็บแสดง (กันกระพริบ) — อ่านค่าที่ผู้ใช้เลือกไว้ ถ้าอ่านไม่ได้ใช้ตามระบบ
const themeScript = `try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="th" suppressHydrationWarning className="h-full antialiased">
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-full flex flex-col">
        <Nav />
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 pb-12 pt-4 sm:px-6">{children}</main>
      </body>
    </html>
  );
}
