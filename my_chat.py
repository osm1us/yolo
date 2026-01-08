import argparse
import getpass
import glob
import html
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


def _safe_import_new_sdk():
    try:
        from google import genai

        return genai
    except Exception:
        return None


def _safe_import_legacy_sdk():
    try:
        import google.generativeai as genai

        return genai
    except Exception:
        return None


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _get_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if key:
        return key
    return getpass.getpass("GEMINI_API_KEY: ")


def _looks_like_model_error(msg: str) -> bool:
    low = msg.lower()
    return (
        "not found" in low
        or "not supported" in low
        or "model" in low
        and "generatecontent" in low
    )


def _looks_like_quota_error(msg: str) -> bool:
    low = msg.lower()
    return (
        "quota" in low
        or "resource_exhausted" in low
        or "rate limit" in low
        or "too many requests" in low
        or "429" in low
    )


def _normalize_new_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    if n.startswith("models/"):
        return n
    if "/" in n:
        return n
    return f"models/{n}"


def _pick_best_model_name(model_names: Iterable[str], preferred_hint: str) -> str:
    names = [n for n in model_names if n]
    lowered = [(n, n.lower()) for n in names]

    def find_by_all_terms(*terms: str) -> Optional[str]:
        termsl = [t.lower() for t in terms if t]
        for original, low in lowered:
            if all(t in low for t in termsl):
                return original
        return None

    for candidate in (
        find_by_all_terms("gemini-3", "flash"),
        find_by_all_terms("gemini-3", "pro"),
        find_by_all_terms("gemini-3", "pro", "preview"),
        find_by_all_terms(preferred_hint, "flash"),
        find_by_all_terms(preferred_hint, "pro"),
        find_by_all_terms("gemini-2.5", "pro"),
        find_by_all_terms("gemini-2.5", "flash"),
        find_by_all_terms("gemini-2.0", "flash"),
    ):
        if candidate:
            return candidate

    return names[0] if names else "gemini-2.5-flash"


def _get_profile_system_instruction(profile: str) -> str:
    p = (profile or "").strip().lower()
    if p == "code":
        return "\n".join(
            [
                "Ты — сеньор-разработчик экспертного уровня, умеющий писать чистый, производительный и поддерживаемый код.",
                "Ты — мой интеллектуальный союзник, объясняющий сложное простыми словами.",
                "Твоя задача — делать код безопасным, стабильным и современным, соблюдая лучшие практики и принципы SOLID, DRY, KISS, YAGNI.",
                "",
                "Формат ответа:",
                "- Сначала короткий план.",
                "- По умолчанию: если пользователь прислал фрагмент кода/ячейку — верни полный обновлённый блок кода (готовый к вставке).",
                "- Patch/diff давай только если пользователь явно попросил diff/patch или если правка относится к файлам проекта.",
                "- Затем короткий список проверок (smoke/unit), если участок критичный.",
                "- Если не хватает контекста — попроси добавить файлы командами /add или /adddir.",
                "- Не придумывай содержимое файлов.",
                "",
                "Качество и чистота:",
                "- Код должен быть лаконичным, читаемым, поддерживаемым.",
                "- Исключай грязные хаки, но избегай чрезмерной усложнённости.",
                "- Следуй современным best practices для конкретного языка.",
                "",
                "Безопасность:",
                "- Проверяй на SQL/NoSQL инъекции, XSS, CSRF, race conditions, небезопасную работу с файловой системой.",
                "- Предлагай безопасные альтернативы, если есть сомнения.",
                "",
                "Стабильность и архитектура:",
                "- Оцени проект целиком перед исправлениями: зависимости, паттерны, тесты.",
                "- Не ломай архитектуру на лету.",
                "",
                "Интерактивность:",
                "- Если задача не ясна, задавай уточняющие вопросы.",
                "- Давай несколько вариантов решения с плюсами/минусами, если уместно.",
            ]
        )
    return "Отвечай кратко и по делу."


def _looks_like_text_file(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            head = f.read(2048)
        return b"\x00" not in head
    except Exception:
        return False


def _is_ignored_path(p: Path) -> bool:
    ignored_dirs = {
        ".git",
        ".hg",
        ".svn",
        ".ipynb_checkpoints",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "dist",
        "build",
    }
    parts = set(p.parts)
    if parts & ignored_dirs:
        return True
    low = p.name.lower()
    return low.endswith((
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".whl",
        ".pyc",
        ".pyo",
        ".pyd",
    ))


@dataclass
class ContextStore:
    max_total_chars: int = 120_000
    max_file_chars: int = 40_000
    root_dir: str = ""

    def __post_init__(self) -> None:
        self._files: dict[str, str] = {}
        self.root_dir = self.root_dir or str(Path.cwd())

    def clear(self) -> None:
        self._files.clear()

    def drop(self, path: str) -> bool:
        key = self._norm_key(path)
        if key in self._files:
            del self._files[key]
            return True
        return False

    def add_file(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(self.root_dir) / p
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))
        if _is_ignored_path(p):
            raise ValueError("ignored path")
        if not _looks_like_text_file(p):
            raise ValueError("binary-like file")

        try:
            size = p.stat().st_size
            if size > self.max_file_chars * 8:
                raise ValueError("file too large")
        except Exception:
            pass

        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > self.max_file_chars:
            text = text[-self.max_file_chars :]
        self._files[self._norm_key(str(p))] = text
        self._trim_to_budget()
        return self._norm_key(str(p))

    def add_dir(self, dir_path: str, pattern: str = "**/*") -> list[str]:
        base = Path(dir_path).expanduser()
        if not base.is_absolute():
            base = Path(self.root_dir) / base
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(str(base))
        paths = glob.glob(str(base / pattern), recursive=True)
        added: list[str] = []
        for p in paths:
            pp = Path(p)
            if not pp.is_file():
                continue
            if _is_ignored_path(pp):
                continue
            try:
                added.append(self.add_file(str(pp)))
            except Exception:
                continue
        return added

    def refresh(self) -> str:
        keys = list(self._files.keys())
        kept = 0
        dropped = 0
        for k in keys:
            p = Path(k)
            if not p.exists() or not p.is_file() or _is_ignored_path(p) or not _looks_like_text_file(p):
                del self._files[k]
                dropped += 1
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                if len(text) > self.max_file_chars:
                    text = text[-self.max_file_chars :]
                self._files[k] = text
                kept += 1
            except Exception:
                del self._files[k]
                dropped += 1
        self._trim_to_budget()
        return f"ok: kept={kept} dropped={dropped}"

    def summary(self) -> str:
        items = [(k, len(v)) for k, v in self._files.items()]
        total = sum(n for _, n in items)
        lines = [f"files={len(items)} total_chars={total} budget={self.max_total_chars}"]
        for k, n in sorted(items, key=lambda x: x[0].lower()):
            lines.append(f"- {k} ({n} chars)")
        return "\n".join(lines)

    def render(self) -> str:
        if not self._files:
            return ""
        parts: list[str] = []
        for k in sorted(self._files.keys(), key=lambda x: x.lower()):
            parts.append(f"<file path=\"{k}\">\n{self._files[k]}\n</file>")
        return "\n\n".join(parts)

    def _norm_key(self, path: str) -> str:
        return str(Path(path).expanduser().resolve())

    def _trim_to_budget(self) -> None:
        total = sum(len(v) for v in self._files.values())
        if total <= self.max_total_chars:
            return
        keys = sorted(self._files.keys(), key=lambda k: len(self._files[k]), reverse=True)
        for k in keys:
            if total <= self.max_total_chars:
                break
            removed = len(self._files[k])
            del self._files[k]
            total -= removed


def _handle_command(
    command_text: str,
    engine: "ChatEngine",
    ctx: ContextStore,
    profile: str,
) -> str:
    raw = (command_text or "").strip()
    if not raw.startswith("/"):
        raise ValueError("not a command")

    parts = raw.split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in {"/help", "/h"}:
        return "\n".join(
            [
                "/help",
                "/status",
                "/models",
                "/pwd",
                "/cd <dir>",
                "/ctx",
                "/add <path>",
                "/adddir <dir> [glob]",
                "/refresh",
                "/use <model>",
                "/drop <path>",
                "/clear",
                "/prompt",
            ]
        )

    if cmd == "/status":
        return "\n".join(
            [
                f"backend={engine.backend}",
                f"model={engine.model_name}",
                f"profile={profile}",
                f"system_prompt_chars={len(engine.system_instruction or '')}",
                ctx.summary(),
            ]
        )

    if cmd == "/prompt":
        return engine.system_instruction or ""

    if cmd == "/models":
        models = list_models(api_key=engine.api_key)
        if not models:
            return "models list is not available"
        return "\n".join(models)

    if cmd == "/ctx":
        return ctx.summary()

    if cmd == "/pwd":
        return ctx.root_dir

    if cmd == "/cd":
        if not args:
            return "usage: /cd <dir>"
        p = Path(" ".join(args)).expanduser()
        if not p.is_absolute():
            p = Path(ctx.root_dir) / p
        if not p.exists() or not p.is_dir():
            return "not a dir"
        ctx.root_dir = str(p.resolve())
        return ctx.root_dir

    if cmd == "/clear":
        ctx.clear()
        return "ok"

    if cmd == "/drop":
        if not args:
            return "usage: /drop <path>"
        return "ok" if ctx.drop(" ".join(args)) else "not found"

    if cmd == "/add":
        if not args:
            return "usage: /add <path>"
        key = ctx.add_file(" ".join(args))
        return f"ok: {key}"

    if cmd == "/adddir":
        d = args[0] if args else "."
        pattern = args[1] if len(args) > 1 else "**/*"
        added = ctx.add_dir(d, pattern=pattern)
        return f"ok: added={len(added)}"

    if cmd == "/refresh":
        return ctx.refresh()

    if cmd == "/use":
        if not args:
            return "usage: /use <model>"
        model = " ".join(args).strip()
        if not model:
            return "usage: /use <model>"
        engine.set_model(model)
        return f"ok: model={engine.model_name}"

    return "unknown command"


@dataclass
class ChatEngine:
    api_key: str
    model_name: Optional[str] = None
    prefer_new_sdk: bool = True
    system_instruction: str = ""
    allow_legacy_fallback: bool = True

    def __post_init__(self) -> None:
        self.backend = None
        self._new_genai = _safe_import_new_sdk() if self.prefer_new_sdk else None
        self._legacy_genai = None

        if self._new_genai is not None:
            self.backend = "google-genai"
            self.client = self._new_genai.Client(api_key=self.api_key)
            chosen = self.model_name or self._pick_new_model(preferred_hint="flash")
            self.set_model(chosen)
            return

        if self.allow_legacy_fallback:
            self._legacy_genai = _safe_import_legacy_sdk()

        if self._legacy_genai is not None:
            self.backend = "google-generativeai"
            self._legacy_genai.configure(api_key=self.api_key)
            self.model_name = self.model_name or "gemini-1.5-flash"
            self.model = self._legacy_genai.GenerativeModel(self.model_name)
            self.chat = self.model.start_chat(history=[])
            return

        raise RuntimeError(
            "No Gemini SDK is available. Install either 'google-genai' (preferred) or 'google-generativeai'."
        )

    def _pick_new_model(self, preferred_hint: str) -> str:
        try:
            names: list[str] = []
            for m in self.client.models.list():
                actions = getattr(m, "supported_actions", None)
                name = getattr(m, "name", None)
                if not name or not actions:
                    continue
                if "generateContent" in actions:
                    names.append(name)
            return _pick_best_model_name(names, preferred_hint=preferred_hint)
        except Exception:
            return "gemini-2.5-flash"

    def set_model(self, model_name: str) -> None:
        if not model_name:
            return
        if self.backend == "google-genai":
            self.model_name = _normalize_new_model_name(model_name)
        else:
            self.model_name = model_name
        if self.backend == "google-genai":
            self.chat = self.client.chats.create(model=self.model_name)
            return
        if self.backend == "google-generativeai":
            self.model = self._legacy_genai.GenerativeModel(self.model_name)
            self.chat = self.model.start_chat(history=[])
            return

    def send(self, message: str) -> str:
        if not message:
            return ""

        try:
            if self.backend == "google-genai":
                resp = self.chat.send_message(message=message)
                return getattr(resp, "text", "")

            resp = self.chat.send_message(message)
            return getattr(resp, "text", "")
        except Exception as e:
            if self.backend == "google-genai":
                msg = str(e)
                if _looks_like_quota_error(msg):
                    alt = self._pick_new_model(preferred_hint="flash")
                    if alt and alt != self.model_name:
                        self.set_model(alt)
                        resp = self.chat.send_message(message=message)
                        return getattr(resp, "text", "")

                if _looks_like_model_error(msg):
                    alt = self._pick_new_model(preferred_hint="flash")
                    if alt and alt != self.model_name:
                        self.set_model(alt)
                        resp = self.chat.send_message(message=message)
                        return getattr(resp, "text", "")
            raise


@dataclass
class AssistantSession:
    engine: ChatEngine
    profile: str = "default"
    context: ContextStore = field(default_factory=ContextStore)

    def set_system_prompt(self, prompt: str) -> None:
        self.engine.system_instruction = prompt

    def build_message(self, user_text: str) -> str:
        system = self.engine.system_instruction
        ctx = self.context.render()
        if ctx:
            return f"SYSTEM:\n{system}\n\nCONTEXT:\n{ctx}\n\nUSER:\n{user_text}"
        return f"SYSTEM:\n{system}\n\nUSER:\n{user_text}"

    def handle_user_input(self, user_text: str) -> str:
        t = (user_text or "").strip()
        if t.startswith("/"):
            return _handle_command(t, self.engine, self.context, self.profile)
        return self.engine.send(self.build_message(t))


class WidgetChat:
    def __init__(self, session: AssistantSession):
        import ipywidgets as widgets
        from IPython.display import display

        self.session = session
        self.model_name = session.engine.model_name

        self._widgets = widgets
        self._display = display
        self._in_flight = False

        self.header = widgets.HTML()
        self.chat_box = widgets.VBox(
            layout=widgets.Layout(width="100%", height="420px", overflow_y="auto")
        )

        self.text_input = widgets.Textarea(
            placeholder="Напиши вопрос или скопируй ошибку сюда...",
            layout=widgets.Layout(width="100%", height="100px"),
        )
        self.send_btn = widgets.Button(
            description="Спросить",
            button_style="info",
            layout=widgets.Layout(width="150px"),
        )
        self.clear_btn = widgets.Button(description="Очистить чат", button_style="warning")

        self.btn_status = widgets.Button(description="Status", layout=widgets.Layout(width="110px"))
        self.btn_ctx = widgets.Button(description="Context", layout=widgets.Layout(width="110px"))
        self.btn_prompt = widgets.Button(description="Prompt", layout=widgets.Layout(width="110px"))
        self.btn_models = widgets.Button(description="Models", layout=widgets.Layout(width="110px"))
        self.btn_help = widgets.Button(description="Help", layout=widgets.Layout(width="110px"))

        self.panel_status = widgets.Textarea(layout=widgets.Layout(width="100%", height="260px"), disabled=True)
        self.panel_ctx = widgets.Textarea(layout=widgets.Layout(width="100%", height="260px"), disabled=True)
        self.panel_prompt = widgets.Textarea(layout=widgets.Layout(width="100%", height="260px"), disabled=True)
        self.panel_models = widgets.Textarea(layout=widgets.Layout(width="100%", height="260px"), disabled=True)
        self.panel_help = widgets.Textarea(layout=widgets.Layout(width="100%", height="260px"), disabled=True)

        self.tabs = widgets.Tab(children=[self.panel_status, self.panel_ctx, self.panel_prompt, self.panel_models, self.panel_help])
        self.tabs.set_title(0, "Status")
        self.tabs.set_title(1, "Context")
        self.tabs.set_title(2, "Prompt")
        self.tabs.set_title(3, "Models")
        self.tabs.set_title(4, "Help")

        self.send_btn.on_click(self.handle_submit)
        self.clear_btn.on_click(self.clear_output)
        self.btn_status.on_click(lambda _b: self._refresh_panel("/status"))
        self.btn_ctx.on_click(lambda _b: self._refresh_panel("/ctx"))
        self.btn_prompt.on_click(lambda _b: self._refresh_panel("/prompt"))
        self.btn_models.on_click(lambda _b: self._refresh_panel("/models"))
        self.btn_help.on_click(lambda _b: self._refresh_panel("/help"))

        self._render_header()
        self._refresh_panel("/status")
        self._refresh_panel("/help")

        display(
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            self.header,
                            self.chat_box,
                            self.text_input,
                            widgets.HBox([self.send_btn, self.clear_btn]),
                        ],
                        layout=widgets.Layout(width="70%"),
                    ),
                    widgets.VBox(
                        [
                            widgets.HBox(
                                [self.btn_status, self.btn_ctx, self.btn_prompt, self.btn_models, self.btn_help],
                                layout=widgets.Layout(justify_content="space-between"),
                            ),
                            self.tabs,
                        ],
                        layout=widgets.Layout(width="30%"),
                    ),
                ],
                layout=widgets.Layout(width="100%"),
            )
        )

    def _render_header(self) -> None:
        backend = self.session.engine.backend
        model = self.session.engine.model_name
        self.header.value = (
            "<div style='display:flex;gap:12px;align-items:center;"
            "padding:8px 10px;border:1px solid #333;border-radius:8px;'>"
            f"<div><b>Backend</b>: {html.escape(str(backend))}</div>"
            f"<div><b>Model</b>: {html.escape(str(model))}</div>"
            "</div>"
        )

    def _append_message(self, role: str, text: str, *, pending: bool = False) -> "object":
        safe = html.escape(text or "")
        if role == "user":
            style = "background:#1f2937;border:1px solid #374151;"
            title = "You"
        else:
            style = "background:#111827;border:1px solid #1f2937;"
            title = "Gemini"

        suffix = " <span style='opacity:0.7'>(...)</span>" if pending else ""
        box = self._widgets.HTML(
            value=(
                "<div style='margin:8px 0;padding:10px;border-radius:10px;" + style + "'>"
                f"<div style='font-weight:700;margin-bottom:6px'>{title}{suffix}</div>"
                f"<div style='white-space:pre-wrap;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;'>{safe}</div>"
                "</div>"
            )
        )
        self.chat_box.children = (*self.chat_box.children, box)
        return box

    def _update_message(self, box: "object", role: str, text: str) -> None:
        safe = html.escape(text or "")
        if role == "user":
            style = "background:#1f2937;border:1px solid #374151;"
            title = "You"
        else:
            style = "background:#111827;border:1px solid #1f2937;"
            title = "Gemini"
        box.value = (
            "<div style='margin:8px 0;padding:10px;border-radius:10px;" + style + "'>"
            f"<div style='font-weight:700;margin-bottom:6px'>{title}</div>"
            f"<div style='white-space:pre-wrap;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;'>{safe}</div>"
            "</div>"
        )

    def _refresh_panel(self, command: str) -> None:
        try:
            text = self.session.handle_user_input(command)
        except Exception as e:
            text = f"ERROR: {e}"

        if command == "/status":
            self.panel_status.value = text
            self.tabs.selected_index = 0
        elif command == "/ctx":
            self.panel_ctx.value = text
            self.tabs.selected_index = 1
        elif command == "/prompt":
            self.panel_prompt.value = text
            self.tabs.selected_index = 2
        elif command == "/models":
            self.panel_models.value = text
            self.tabs.selected_index = 3
        elif command == "/help":
            self.panel_help.value = text
            self.tabs.selected_index = 4

        self._render_header()

    def handle_submit(self, _b):
        if self._in_flight:
            return

        user_msg = self.text_input.value
        if not user_msg:
            return

        self.text_input.value = ""

        self._in_flight = True
        self.send_btn.disabled = True

        self._append_message("user", user_msg)
        pending = self._append_message("assistant", "Gemini думает...", pending=True)

        try:
            text = self.session.handle_user_input(user_msg)
            self.model_name = self.session.engine.model_name
            self._update_message(pending, "assistant", text)
        except Exception as e:
            self._update_message(pending, "assistant", f"Ошибка API: {e}")
        finally:
            self._render_header()
            self.send_btn.disabled = False
            self._in_flight = False

    def clear_output(self, _b):
        self.chat_box.children = ()


def list_models(api_key: Optional[str] = None) -> list[str]:
    key = _get_api_key(api_key)
    new_genai = _safe_import_new_sdk()
    if new_genai is None:
        return []

    client = new_genai.Client(api_key=key)
    models: list[str] = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None)
        name = getattr(m, "name", None)
        if not name or not actions:
            continue
        if "generateContent" in actions:
            models.append(name)
    return sorted(set(models))


def start(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    ui: str = "auto",
    profile: str = "default",
    max_context_chars: int = 120_000,
    system_prompt: Optional[str] = None,
    allow_legacy_fallback: bool = True,
) -> object:
    key = _get_api_key(api_key)

    sys_inst = system_prompt if system_prompt is not None else _get_profile_system_instruction(profile)

    if ui == "cli" or (ui == "auto" and not _is_notebook()):
        engine = ChatEngine(
            api_key=key,
            model_name=model_name,
            system_instruction=sys_inst,
            allow_legacy_fallback=allow_legacy_fallback,
        )
        return start_cli(engine)

    engine = ChatEngine(
        api_key=key,
        model_name=model_name,
        system_instruction=sys_inst,
        allow_legacy_fallback=allow_legacy_fallback,
    )
    session = AssistantSession(engine=engine, profile=profile, context=ContextStore(max_total_chars=max_context_chars))
    return WidgetChat(session)


def start_with_model(api_key: str, model_name: Optional[str] = None) -> object:
    return start(api_key=api_key, model_name=model_name, ui="auto")


def start_cli(engine: Optional[ChatEngine] = None) -> None:
    eng = engine or ChatEngine(api_key=_get_api_key(None))
    session = AssistantSession(engine=eng, profile="default", context=ContextStore())
    print(f"Backend: {eng.backend}")
    print(f"Model: {eng.model_name}")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        try:
            print(session.handle_user_input(msg))
        except Exception as e:
            print(f"Ошибка API: {e}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        for name in list_models(api_key=args.api_key):
            print(name)
        return

    key = _get_api_key(args.api_key)
    sys_inst = args.system_prompt if args.system_prompt is not None else _get_profile_system_instruction(args.profile)
    engine = ChatEngine(
        api_key=key,
        model_name=args.model,
        system_instruction=sys_inst,
    )
    start_cli(engine)


if __name__ == "__main__":
    _main()
