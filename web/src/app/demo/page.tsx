import type { Metadata } from "next";
import DemoView from "@/components/DemoView";

export const metadata: Metadata = { title: "บัญชีเดโม · BTC / Gold Trend" };

export default function Page() {
  return <DemoView />;
}
