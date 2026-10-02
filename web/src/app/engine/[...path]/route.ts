import type { NextRequest } from "next/server";

// proxy ไป FastAPI : อ่านที่อยู่ตอนรันจริง (Docker = http://api:8000)
const API_URL = () => process.env.API_URL ?? "http://localhost:8000";

async function forward(req: NextRequest, path: string[], init: RequestInit) {
  const url = `${API_URL()}/api/${path.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;
  try {
    const r = await fetch(url, { ...init, cache: "no-store", signal: AbortSignal.timeout(60_000) });
    return new Response(r.body, {
      status: r.status,
      headers: { "content-type": r.headers.get("content-type") ?? "application/json", "cache-control": "no-store" },
    });
  } catch {
    return Response.json({ detail: "เชื่อมต่อ engine API ไม่ได้ — เปิด: docker compose up -d api" }, { status: 502 });
  }
}

export async function GET(req: NextRequest, ctx: RouteContext<"/engine/[...path]">) {
  const { path } = await ctx.params;
  return forward(req, path, { method: "GET" });
}

export async function POST(req: NextRequest, ctx: RouteContext<"/engine/[...path]">) {
  const { path } = await ctx.params;
  return forward(req, path, {
    method: "POST",
    headers: { "content-type": req.headers.get("content-type") ?? "application/json" },
    body: await req.text(),
  });
}
