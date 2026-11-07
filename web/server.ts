import { fileURLToPath } from "node:url";
import { join } from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const maxBody = 65_536;
export const maxText = 12_000;

class HttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

type Translation = { seconds: number; truncated: boolean };
export type Translate = (text: string, target: string, signal: AbortSignal, onDelta: (text: string) => void) => Promise<Translation>;

const encoder = new TextEncoder();

export const translate: Translate = async (text, target, signal, onDelta) => {
  signal.throwIfAborted();
  const started = performance.now();
  const child = Bun.spawn({
    cmd: [join(root, ".build/release/plamo"), "--packed", "--stream", "--batch4", "--segment-tokens", "512", "--to", target, "--max-tokens", "2048"],
    cwd: root,
    stdin: new TextEncoder().encode(text),
    stdout: "pipe",
    stderr: "pipe",
  });
  const cancel = () => child.kill();
  signal.addEventListener("abort", cancel, { once: true });
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; child.kill(); }, 180_000);
  try {
    const stderr = new Response(child.stderr).text();
    const decoder = new TextDecoder();
    const drain = (async () => {
      const reader = child.stdout.getReader();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        if (chunk) onDelta(chunk);
      }
      const tail = decoder.decode();
      if (tail) onDelta(tail);
    })();
    const [code, diagnostic] = await Promise.all([child.exited, stderr, drain]).then(([exit, error]) => [exit, error] as const);
    if (signal.aborted) throw new HttpError(499, "翻訳を中止しました。");
    if (timedOut) throw new HttpError(504, "翻訳が時間内に終わりませんでした。原文を短くして再度お試しください。");
    if (code !== 0) {
      if (/Input (plus output limit exceeds context|exceeds context)/.test(diagnostic)) {
        throw new HttpError(422, "原文がモデルの上限を超えています。原文を短くしてください。");
      }
      throw new HttpError(500, "翻訳エンジンを実行できませんでした。モデルファイルを確認してください。");
    }
    const run = JSON.parse(diagnostic).runs[0];
    if (!["stop", "length", "repeat"].includes(run.stop_reason)) throw new Error("Invalid runtime response");
    return { seconds: (performance.now()-started)/1000, truncated: run.stop_reason !== "stop" };
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener("abort", cancel);
  }
};

const assets = new Map([
  ["/", "index.html"], ["/app.js", "app.js"], ["/app.css", "app.css"],
  ...["global.css", "button.css", "textarea.css", "select.css", "form-control-label.css"].map(name => [`/vendor/dads/${name}`, `vendor/dads/${name}`] as [string, string]),
]);
const headers = {
  "Cache-Control": "no-store",
  "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
};

export function createHandler(origin: string, run: Translate = translate) {
  let busy = false;
  const json = (value: unknown, status = 200) => Response.json(value, { status, headers });
  return async (request: Request): Promise<Response> => {
    const url = new URL(request.url);
    if (url.origin !== origin) return json({ error: "この端末のURLからアクセスしてください。" }, 403);
    if (["GET", "HEAD"].includes(request.method) && assets.has(url.pathname)) {
      const file = Bun.file(join(root, "web", assets.get(url.pathname)!));
      return new Response(request.method === "HEAD" ? null : file, { headers: { ...headers, "Content-Type": file.type } });
    }
    if (url.pathname !== "/api/translate") return json({ error: "ページが見つかりません。" }, 404);
    if (request.method !== "POST") return json({ error: "POSTを使用してください。" }, 405);
    if (request.headers.get("Origin") !== origin) return json({ error: "別のサイトからは翻訳できません。" }, 403);
    if (request.headers.get("Content-Type")?.split(";")[0].trim() !== "application/json") {
      return json({ error: "JSON形式で送信してください。" }, 415);
    }
    let value: unknown;
    try {
      if (Number(request.headers.get("Content-Length")) > maxBody) throw new HttpError(413, "送信内容が大きすぎます。");
      const body = await request.arrayBuffer();
      if (body.byteLength > maxBody) throw new HttpError(413, "送信内容が大きすぎます。");
      value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(body));
    } catch (error) {
      return json({ error: error instanceof HttpError ? error.message : "送信内容を読み取れませんでした。" }, error instanceof HttpError ? error.status : 400);
    }
    if (!value || typeof value !== "object" || !("text" in value) || typeof value.text !== "string" ||
        !("target" in value) || !["ja", "en"].includes(value.target as string)) {
      return json({ error: "原文と翻訳先の言語を確認してください。" }, 400);
    }
    const { text, target } = value as { text: string; target: string };
    if (!text.trim()) return json({ error: "原文を入力してください。" }, 400);
    if (text.length > maxText) return json({ error: "原文は12,000文字以内で入力してください。" }, 413);
    if (text.includes("<|plamo:") || text.includes("\uee00")) return json({ error: "原文にモデルの予約文字が含まれています。" }, 400);
    if (request.signal.aborted) return json({ error: "翻訳を中止しました。" }, 499);
    if (busy) return json({ error: "別の翻訳を実行中です。完了後に再度お試しください。" }, 409);
    busy = true;
    // Kill the engine when the response stream is cancelled, not only when the request aborts.
    const abort = new AbortController();
    const forward = () => abort.abort();
    if (request.signal.aborted) abort.abort();
    else request.signal.addEventListener("abort", forward, { once: true });
    const body = new ReadableStream({
      async start(controller) {
        const send = (value: unknown) => controller.enqueue(encoder.encode(JSON.stringify(value)+"\n"));
        const delta = (chunk: string) => { try { send({ type: "delta", text: chunk }); } catch { abort.abort(); } };
        try {
          const result = await run(text, target, abort.signal, delta);
          send({ type: "done", seconds: result.seconds, truncated: result.truncated });
        } catch (error) {
          const status = error instanceof HttpError ? error.status : 500;
          const message = error instanceof HttpError ? error.message : "翻訳に失敗しました。再度お試しください。";
          try { send({ type: "error", status, message }); } catch {}
        } finally {
          busy = false;
          request.signal.removeEventListener("abort", forward);
          try { controller.close(); } catch {}
        }
      },
      cancel() { abort.abort(); },
    });
    return new Response(body, { headers: { ...headers, "Content-Type": "application/x-ndjson; charset=utf-8" } });
  };
}

if (import.meta.main) {
  for (const file of [".build/release/plamo", "models/plamo/model.safetensors", "models/plamo/model.p12", "models/plamo/config.json", "models/plamo/tokenizer.jsonl"]) {
    if (!await Bun.file(join(root, file)).exists()) throw new Error(`${file} がありません。README.mdのモデル準備手順を実行してください。`);
  }
  const port = Number(Bun.env.PORT ?? 3000);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("PORT must be in 1...65535");
  const origin = `http://127.0.0.1:${port}`;
  Bun.serve({ hostname: "127.0.0.1", port, idleTimeout: 255, maxRequestBodySize: maxBody, fetch: createHandler(origin) });
  console.log(`翻訳ページ: ${origin}`);
}
