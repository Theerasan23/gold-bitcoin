import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Docker : สร้าง .next/standalone (server.js + ไฟล์ที่ต้องใช้เท่านั้น)
  output: "standalone",
};

export default nextConfig;
