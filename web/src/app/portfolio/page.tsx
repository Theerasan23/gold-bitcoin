import type { Metadata } from "next";
import PortfolioView from "@/components/PortfolioView";

export const metadata: Metadata = { title: "พอร์ต · BTC / Gold Trend" };

export default function Page() {
  return <PortfolioView />;
}
