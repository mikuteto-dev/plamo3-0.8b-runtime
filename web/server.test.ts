import { describe, test, expect, spyOn } from "bun:test";
import { createHandler, maxText, translate } from "./server";

const origin = "http://127.0.0.1:3000";
const done = { type: "done", seconds: 0.5, truncated: false } as const;
function request(value: unknown = { text: "原文", target: "en" }, init: RequestInit = {}) {
  return new Request(`${origin}/api/translate`, {
    method: "POST", headers: { Origin: origin, "Content-Type": "application/json" }, body: JSON.stringify(value), ...init,
  });
}
async function events(response: Response) {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const out: unknown[] = [];
  for (;;) {
    const { value, done: finished } = await reader.read();
    if (finished) break;
    buffer += decoder.decode(value, { stream: true });
    let index: number;
    while ((index = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, index).trim();
      buffer = buffer.slice(index + 1);
      if (line) out.push(JSON.parse(line));
    }
  }
  return out as Record<string, string>[];
}
const deltas = (list: Record<string, string>[]) => list.filter(event => event.type === "delta").map(event => event.text).join("");

describe("local translation server", () => {
  test("uses ordered four-segment GPU batching without passing source as an argument", async () => {
    const calls: { cmd: string[]; stdin: Uint8Array }[] = [];
    const spawn = spyOn(Bun, "spawn").mockImplementation(options => {
      calls.push(options as { cmd: string[]; stdin: Uint8Array });
      return {
        stdout: new Response("first\nsecond").body!,
        stderr: new Response(JSON.stringify({ runs: [{ stop_reason: "stop" }] })).body!,
        exited: Promise.resolve(0), kill() {},
      } as unknown as ReturnType<typeof Bun.spawn>;
    });
    try {
      let output = "";
      const text = "  first\nsecond  ";
      const result = await translate(text, "en", new AbortController().signal, chunk => { output += chunk; });
      expect(calls).toHaveLength(1);
      expect(calls[0].cmd[0].endsWith("/.build/release/plamo")).toBe(true);
      expect(calls[0].cmd.slice(1)).toEqual(["--packed", "--stream", "--batch4", "--segment-tokens", "512", "--to", "en", "--max-tokens", "2048"]);
      expect(new TextDecoder().decode(calls[0].stdin)).toBe(text);
      expect(output).toBe("first\nsecond");
      expect(result.truncated).toBe(false);
    } finally {
      spawn.mockRestore();
    }
  });

  test("serves only page assets with local-only security headers", async () => {
    const handle = createHandler(origin);
    for (const path of ["/", "/app.js", "/app.css", "/vendor/dads/global.css"]) {
      const response = await handle(new Request(origin+path));
      expect(response.status).toBe(200);
      expect(await response.text()).not.toBe("");
      expect(response.headers.get("Content-Security-Policy")).toContain("default-src 'self'");
      expect(response.headers.get("Cache-Control")).toBe("no-store");
    }
    const head = await handle(new Request(origin, { method: "HEAD" }));
    expect(head.status).toBe(200);
    expect(head.headers.get("Content-Type")).toContain("text/html");
    expect(await head.text()).toBe("");
    for (const path of ["/models/plamo/model.p12", "/server.ts", "/.env", "/vendor/dads/../../server.ts"]) {
      expect((await handle(new Request(origin+path))).status).toBe(404);
    }
    expect((await handle(new Request("http://attacker.example/"))).status).toBe(403);
  });

  test("streams deltas and passes text through without shell interpretation or whitespace loss", async () => {
    const values: unknown[] = [];
    const handle = createHandler(origin, async (text, target, _signal, onDelta) => {
      values.push([text, target]);
      onDelta("Trans");
      onDelta("lation");
      return done;
    });
    const text = "  $(echo nope); <script>alert(1)</script>\n原文  ";
    const response = await handle(request({ text, target: "ja" }));
    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toContain("application/x-ndjson");
    const list = await events(response);
    expect(deltas(list)).toBe("Translation");
    expect(list.at(-1)).toEqual(done);
    expect(values).toEqual([[text, "ja"]]);
  });

  test("delivers each delta before the translation finishes", async () => {
    const gate = Promise.withResolvers<void>();
    const handle = createHandler(origin, async (_text, _target, _signal, onDelta) => {
      onDelta("early");
      await gate.promise;
      return done;
    });
    const reader = (await handle(request())).body!.getReader();
    const first = await reader.read();
    expect(new TextDecoder().decode(first.value)).toBe('{"type":"delta","text":"early"}\n');
    gate.resolve();
    await reader.cancel();
  });

  test("rejects foreign and absent origins, wrong methods and content types", async () => {
    const handle = createHandler(origin, async () => { throw new Error("Must not run"); });
    for (const headers of [{ "Content-Type": "application/json" }, { Origin: "https://attacker.example", "Content-Type": "application/json" }]) {
      expect((await handle(request(undefined, { headers }))).status).toBe(403);
    }
    expect((await handle(new Request(origin+"/api/translate"))).status).toBe(405);
    expect((await handle(request(undefined, { headers: { Origin: origin, "Content-Type": "text/plain" } }))).status).toBe(415);
  });

  test("validates JSON, shape, languages, empty input and reserved delimiters", async () => {
    const handle = createHandler(origin, async () => { throw new Error("Must not run"); });
    for (const value of [null, [], {}, { text: 1, target: "en" }, { text: "x", target: "../en" },
      { text: "", target: "en" }, { text: " \n\t", target: "en" }, { text: "<|plamo:tag|>", target: "en" }, { text: "a\uee00b", target: "en" }]) {
      expect((await handle(request(value))).status).toBe(400);
    }
    for (const body of ["{", new Uint8Array([0xff, 0xfe])]) {
      expect((await handle(request(undefined, { body }))).status).toBe(400);
    }
  });

  test("bounds request bytes and UTF-16 text length without truncation", async () => {
    const handle = createHandler(origin, async (text, _target, _signal, onDelta) => { onDelta(text); return done; });
    expect((await handle(request({ text: "a".repeat(maxText+1), target: "en" }))).status).toBe(413);
    expect((await handle(request({ text: "a".repeat(maxText), target: "en" }))).status).toBe(200);
    expect((await handle(request(undefined, { body: " ".repeat(65_537) }))).status).toBe(413);
    expect((await handle(request(undefined, { headers: { Origin: origin, "Content-Type": "application/json", "Content-Length": "65537" } }))).status).toBe(413);
  });

  test("runs one translation at a time and releases the slot after completion", async () => {
    const entered = Promise.withResolvers<void>(), release = Promise.withResolvers<void>();
    let calls = 0;
    const handle = createHandler(origin, async () => { calls++; entered.resolve(); await release.promise; return done; });
    const first = handle(request());
    await entered.promise;
    expect((await handle(request())).status).toBe(409);
    expect(calls).toBe(1);
    release.resolve();
    expect((await events(await first)).at(-1)).toEqual(done);
    expect((await handle(request())).status).toBe(200);
    expect(calls).toBe(2);
  });

  test("reports failures without exposing internals and releases the slot", async () => {
    let calls = 0;
    const handle = createHandler(origin, async () => { if (++calls === 1) throw new Error("private internal details"); return done; });
    const failure = await handle(request());
    expect(failure.status).toBe(200);
    const list = await events(failure);
    expect(list).toEqual([{ type: "error", status: 500, message: "翻訳に失敗しました。再度お試しください。" }]);
    expect(JSON.stringify(list)).not.toContain("private internal details");
    expect((await handle(request())).status).toBe(200);
  });

  test("releases the slot when the response stream is cancelled", async () => {
    let calls = 0;
    const aborted = Promise.withResolvers<void>();
    const handle = createHandler(origin, async (_text, _target, signal, onDelta) => {
      if (++calls === 1) {
        signal.addEventListener("abort", () => aborted.resolve(), { once: true });
        onDelta("partial");
        await aborted.promise;
        throw new DOMException("Aborted", "AbortError");
      }
      return done;
    });
    const reader = (await handle(request())).body!.getReader();
    await reader.read();
    await reader.cancel();
    await aborted.promise;
    let response: Response | undefined;
    for (let attempt = 0; attempt < 40; attempt++) {
      response = await handle(request());
      if (response.status === 200) break;
      await Bun.sleep(5);
    }
    expect(response?.status).toBe(200);
    expect(calls).toBe(2);
  });

  test("does not run a cancelled request", async () => {
    const controller = new AbortController(); controller.abort();
    const handle = createHandler(origin, async () => { throw new Error("Must not run"); });
    expect((await handle(request(undefined, { signal: controller.signal }))).status).toBe(499);
  });

  test("marks truncated output in the final event", async () => {
    const handle = createHandler(origin, async (_text, _target, _signal, onDelta) => { onDelta("Partial"); return { seconds: 1, truncated: true }; });
    const list = await events(await handle(request()));
    expect(deltas(list)).toBe("Partial");
    expect(list.at(-1)).toEqual({ type: "done", seconds: 1, truncated: true });
  });
});
