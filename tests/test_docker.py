"""Static checks of the Dockerfile and .dockerignore: no Docker needed to run them."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def instructions() -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) pairs, with comments dropped and `\\` continuations joined."""
    text = (ROOT / "Dockerfile").read_text()
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    joined = re.sub(r"\\\n\s*", " ", "\n".join(lines))
    return [
        (parts[0].upper(), parts[1] if len(parts) > 1 else "")
        for line in joined.splitlines()
        if (parts := line.strip().split(maxsplit=1))
    ]


def runtime_stage() -> list[tuple[str, str]]:
    steps = instructions()
    last_from = max(i for i, (op, _) in enumerate(steps) if op == "FROM")
    return steps[last_from:]


def first(steps: list[tuple[str, str]], op: str, pattern: str) -> int:
    for i, (o, args) in enumerate(steps):
        if o == op and re.search(pattern, args):
            return i
    raise AssertionError(f"no {op} matching {pattern!r} in the Dockerfile")


def ignored(path: str) -> bool:
    """Approximates Docker's .dockerignore rules: the last matching pattern wins, `!` re-includes,
    and a pattern that matches a parent folder applies to everything inside it."""

    def to_regex(pattern: str) -> str:
        out = re.escape(pattern.strip("/"))
        out = out.replace(r"\*\*/", "(.*/)?").replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
        return out.replace(r"\?", "[^/]")

    parts = path.split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    result = False
    for raw in (ROOT / ".dockerignore").read_text().splitlines():
        pattern = raw.strip()
        if not pattern or pattern.startswith("#"):
            continue
        negate = pattern.startswith("!")
        regex = to_regex(pattern.removeprefix("!"))
        if any(re.fullmatch(regex, p) for p in prefixes):
            result = not negate
    return result


# --- provided ---


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (".env", True),
        (".venv/bin/python", True),
        ("tests/test_api.py", True),
        ("logs/rag_answers.jsonl", True),
        ("src/telco_support_agent/__pycache__/api.cpython-312.pyc", True),
        ("src/telco_support_agent/api.py", False),
        ("uv.lock", False),
        ("data/chroma/chroma.sqlite3", False),
    ],
)
def test_dockerignore_keeps_secrets_and_junk_out(path, expected):
    assert ignored(path) is expected


def test_no_secrets_baked_into_the_image():
    for op, args in instructions():
        if op in {"COPY", "ADD"}:
            assert ".env" not in args.split(), f"{op} {args}"
        if op in {"ENV", "ARG"}:
            assert "API_KEY" not in args, f"{op} {args}: pass secrets at run time"


def test_base_images_are_pinned():
    for op, args in instructions():
        if op == "FROM":
            image = args.split()[0]
            assert ":" in image and not image.endswith(":latest"), image


# --- your Task 1: dependencies before code ---


def test_dependencies_install_before_the_code_is_copied():
    steps = instructions()
    lockfile = first(steps, "COPY", r"\buv\.lock\b")
    deps = first(steps, "RUN", r"uv sync .*--no-install-project")
    code = first(steps, "COPY", r"^src\b")
    project = first(steps, "RUN", r"uv sync .*--no-editable")
    assert lockfile < deps < code < project
    workdir = first(steps, "WORKDIR", r"^/app$")
    assert workdir < lockfile, "the runtime stage copies /app/.venv, so the builder works in /app"
    assert "src" not in steps[lockfile][1], "copy src/ only after installing dependencies"


def test_installs_are_locked_and_skip_dev_tools():
    syncs = [args for op, args in instructions() if op == "RUN" and "uv sync" in args]
    assert len(syncs) == 2
    assert all("--locked" in s and "--no-dev" in s for s in syncs)


# --- your Task 2: run as non-root, reachable from outside ---


def test_runs_as_a_non_root_user():
    steps = runtime_stage()
    users = [args.strip() for op, args in steps if op == "USER"]
    assert users and users[-1] not in {"root", "0"}
    assert first(steps, "USER", r".") < first(steps, "CMD", r".")


def test_cmd_is_exec_form_and_listens_on_all_interfaces():
    cmds = [args for op, args in runtime_stage() if op == "CMD"]
    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd.startswith("["), "use exec form so uvicorn gets Docker's stop signal"
    assert "telco_support_agent.api:app" in cmd
    assert "0.0.0.0" in cmd and "8000" in cmd
