const form = document.querySelector("#translate-form");
const source = document.querySelector("#source");
const target = document.querySelector("#target");
const result = document.querySelector("#result");
const submit = document.querySelector("#submit");
const cancel = document.querySelector("#cancel");
const clear = document.querySelector("#clear");
const copy = document.querySelector("#copy");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const warning = document.querySelector("#warning");
const inputError = document.querySelector("#input-error");
const count = document.querySelector("#count");
let active = null;
let stream = null;

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const segmenter = typeof Intl.Segmenter === "function" ? new Intl.Segmenter("ja", { granularity: "grapheme" }) : null;

function graphemes(text) {
  if (!segmenter) return [...text];
  const clusters = [];
  for (const part of segmenter.segment(text)) clusters.push(part.segment);
  return clusters;
}

function resetResult() {
  if (stream?.raf) cancelAnimationFrame(stream.raf);
  stream = null;
  result.textContent = "";
  result.scrollTop = 0;
  result.removeAttribute("lang");
  result.setAttribute("aria-busy", "false");
  copy.disabled = true;
  status.textContent = "";
  error.hidden = true;
  warning.hidden = true;
}

function validate() {
  const message = source.value.length > 12000 ? "原文は12,000文字以内で入力してください。"
    : !source.value.trim() ? "原文を入力してください。" : "";
  inputError.textContent = message;
  inputError.hidden = !message;
  source.setAttribute("aria-invalid", String(Boolean(message)));
  return !message;
}

function updateCount() {
  count.textContent = `${source.value.length.toLocaleString("ja-JP")} / 12,000文字`;
  count.toggleAttribute("data-exceeded", source.value.length > 12000);
  if (source.getAttribute("aria-invalid") === "true") validate();
}

function setBusy(busy) {
  for (const field of [source, target, submit, clear]) field.disabled = busy;
  cancel.hidden = !busy;
  submit.textContent = busy ? "翻訳中…" : "翻訳する";
  result.setAttribute("aria-busy", String(busy));
  copy.disabled = busy || !result.textContent;
}

// Streaming reveal: tokens arrive in bursts, so characters are drained on animation
// frames at an adaptive rate. Each new cluster fades in as a short wave.
function prepareStream() {
  result.textContent = "";
  const text = document.createElement("span");
  text.className = "stream-text";
  const caret = document.createElement("span");
  caret.className = "stream-caret";
  caret.setAttribute("aria-hidden", "true");
  const placeholder = document.createElement("span");
  placeholder.className = "stream-placeholder";
  placeholder.setAttribute("aria-hidden", "true");
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("i");
    dot.style.animationDelay = `${i * 0.16}s`;
    placeholder.appendChild(dot);
  }
  result.append(text, caret, placeholder);
  result.setAttribute("aria-busy", "true");
  stream = { text, caret, placeholder, queue: "", shown: 0, raf: 0, final: null, onDrained: null };
}

function reveal(text) {
  const a = stream;
  if (a.shown >= 4000 || text.length > 200) {
    a.text.appendChild(document.createTextNode(text));
  } else {
    const fragment = document.createDocumentFragment();
    let index = 0;
    for (const cluster of graphemes(text)) {
      const span = document.createElement("span");
      span.className = "stream-char";
      span.style.animationDelay = `${Math.min(index * 9, 120)}ms`;
      span.textContent = cluster;
      fragment.appendChild(span);
      index++;
    }
    a.text.appendChild(fragment);
  }
  a.shown += text.length;
  if (result.scrollHeight - result.scrollTop - result.clientHeight < 48) result.scrollTop = result.scrollHeight;
}

function drain() {
  const a = stream;
  if (!a || a.queue) return;
  if (!a.final) return;
  a.caret.remove();
  result.setAttribute("aria-busy", "false");
  const seconds = a.final.seconds.toFixed(2);
  if (a.final.truncated) {
    warning.textContent = "訳文が途中で終了しています。原文を短くして再度お試しください。";
    warning.hidden = false;
    status.textContent = `${seconds}秒`;
  } else {
    status.textContent = `翻訳しました。${seconds}秒`;
  }
  const resolve = a.onDrained;
  stream = null;
  resolve?.();
}

function pump() {
  const a = stream;
  if (!a) return;
  a.raf = 0;
  if (!a.queue) return drain();
  const take = Math.max(1, Math.ceil(a.queue.length * 0.22));
  const chunk = a.queue.slice(0, take);
  a.queue = a.queue.slice(take);
  reveal(chunk);
  a.raf = requestAnimationFrame(pump);
}

function pushText(text) {
  const a = stream;
  if (!a || !text) return;
  if (a.placeholder) { a.placeholder.remove(); a.placeholder = null; }
  if (reduced) return reveal(text);
  a.queue += text;
  if (!a.raf) a.raf = requestAnimationFrame(pump);
}

function finishStream(final) {
  return new Promise(resolve => {
    if (!stream) return resolve();
    stream.final = final;
    stream.onDrained = resolve;
    if (stream.queue) {
      if (!stream.raf) stream.raf = requestAnimationFrame(pump);
    } else {
      drain();
    }
  });
}

source.addEventListener("input", () => { updateCount(); resetResult(); });
target.addEventListener("change", resetResult);
clear.addEventListener("click", () => {
  source.value = "";
  source.removeAttribute("aria-invalid");
  inputError.hidden = true;
  updateCount();
  resetResult();
  source.focus();
});
cancel.addEventListener("click", () => active?.abort());
copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(result.textContent);
    status.textContent = "訳文をコピーしました。";
  } catch {
    error.textContent = "コピーできませんでした。訳文を選択してコピーしてください。";
    error.hidden = false;
  }
});
form.addEventListener("submit", async event => {
  event.preventDefault();
  if (active) return;
  if (!validate()) { source.focus(); return; }
  resetResult();
  const controller = new AbortController();
  active = controller;
  const language = target.value;
  setBusy(true);
  status.textContent = "翻訳中です。";
  prepareStream();
  try {
    const response = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: source.value, target: language }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const value = await response.json().catch(() => ({}));
      throw new Error(value.error || "翻訳に失敗しました。");
    }
    if (!response.body) throw new Error("翻訳結果を読み取れませんでした。");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let final = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        if (!line) continue;
        const message = JSON.parse(line);
        if (message.type === "delta" && typeof message.text === "string") pushText(message.text);
        else if (message.type === "done") final = message;
        else if (message.type === "error") throw new Error(message.message || "翻訳に失敗しました。");
      }
    }
    if (!final || !Number.isFinite(final.seconds) || typeof final.truncated !== "boolean") {
      throw new Error("翻訳結果を読み取れませんでした。");
    }
    result.lang = language;
    await finishStream(final);
  } catch (cause) {
    if (stream?.raf) cancelAnimationFrame(stream.raf);
    stream = null;
    result.textContent = "";
    if (controller.signal.aborted) {
      status.textContent = "翻訳を中止しました。";
    } else {
      status.textContent = "";
      error.textContent = cause instanceof TypeError ? "サーバーに接続できませんでした。接続を確認して再度お試しください。" : cause.message;
      error.hidden = false;
    }
  } finally {
    active = null;
    setBusy(false);
    if (controller.signal.aborted) submit.focus();
  }
});
