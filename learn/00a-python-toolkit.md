# Lesson 00a — The Python Engineering Toolkit

> **Goal:** learn the tools that separate "a notebook that works" from "code a team can trust":
> uv, type hints, **Pydantic**, **pytest**, **ruff** and **pre-commit**. Every later lesson uses them.
> **Time:** ~3 hours (1.5 h reading, 1 h building, 30 min drill).

Why this matters for AI Engineer interviews: LLM output is *untrusted, loosely structured text*.
The engineers who do well at this job put strict validation at the boundaries, test what can be
tested deterministically, and automate code quality. These tools are how you do that.

---

## Part A — Setup (~20 min)

```bash
# uv: one fast tool that replaces pip + venv + pyenv + pip-tools
brew install uv
cd ~/Documents/wahab/telco-support-agent
uv python install 3.12        # project-local Python; your system 3.9 stays untouched
uv sync                       # creates .venv, installs deps from pyproject.toml, writes uv.lock

git init                      # local only for now
uv run pre-commit install     # wires the hooks into .git/hooks/pre-commit
```

**What `uv` just did, and why it matters:**
- `pyproject.toml` is the single source of truth for the project: dependencies, tool config (ruff,
  pytest) and metadata. It replaces `requirements.txt` + `setup.py` + `setup.cfg`.
- `uv.lock` pins the **exact** version of every package, including transitive ones. Commit it.
  Same lock = same environment on your laptop, in CI and in Docker. (Recall the classic "works on my
  machine" problem. The lock file is half the fix, and Docker in Lesson 04b is the other half.)
- `uv run <cmd>` runs a command inside the project's virtualenv, so there's no `source .venv/bin/activate`.
- `uv add <pkg>` adds a dependency. `uv add --dev <pkg>` adds a development-only one (like pytest).

---

## Part B — Concepts

### 1. Type hints (5-minute version)

```python
def estimate_cost(input_tokens: int, output_tokens: int, prices: Prices) -> float: ...
data_gb: float | None = None        # "float or None"; the old spelling was Optional[float]
def first[T](items: list[T]) -> T: ...   # generic function (Python 3.12 syntax)
```

Plain Python **ignores type hints at runtime**. They're documentation that tools can check (your
editor, `mypy`, `pyright`). **Pydantic is the exception**: it reads the hints and *enforces* them at
runtime. That's the whole idea behind it.

### 2. Pydantic: validation at the boundaries

**The problem.** Data entering your system from outside is untrusted: HTTP requests, config files,
and, specific to LLM apps, **model output and tool-call arguments**. If you pass a raw `dict` around,
a wrong type or missing key blows up three functions later, far from the cause.

**The idea.** Declare the shape once as a class. Pydantic parses incoming data into it, **coercing
or rejecting** at the boundary. Inside your code you then work with a typed object you can trust.

```python
from pydantic import BaseModel, Field, ValidationError

class Plan(BaseModel):
    name: str
    price_sar: float = Field(ge=0, description="Monthly price in SAR")
    data_gb: float | None = None          # optional, defaults to None

Plan(name="Super 50", price_sar="49.5")   # OK: "49.5" is coerced to 49.5 (lax mode)
Plan(name="Super 50", price_sar=-1)       # ValidationError: greater than or equal to 0
Plan(price_sar=49.5)                      # ValidationError: name, field required
```

The error tells you *which field* failed and *why*. Compare that with a `KeyError` deep in some function.

**The methods you'll use constantly:**

| Method | Does | Where we use it |
|---|---|---|
| `Plan(**kwargs)` / `Plan.model_validate(dict)` | dict → validated object | Loading the plan catalog (Phase 1) |
| `Plan.model_validate_json(str)` | JSON string → object | Parsing LLM output, API bodies |
| `p.model_dump()` / `p.model_dump_json()` | object → dict / JSON | Logging, API responses, eval records |
| `Plan.model_json_schema()` | class → **JSON Schema** | **What the LLM sees** for structured outputs and tools |

**That last row is the key AI-engineering point.** In Lesson 00b, `responses.parse(text_format=UsageProfile)`
turns your class into a JSON Schema and sends it to the model. **Your `Field(description=...)` text
becomes part of the prompt.** Good descriptions ("Monthly data in GB; null if not stated") directly
improve extraction accuracy. FastAPI (04a) uses the same mechanism to generate API docs, and tool
calling (10) uses it for function arguments. You learn it once and use it three times.

**Constraints and custom validators:**

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator

class UsageProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")      # reject unknown keys instead of silently ignoring them

    data_gb: float | None = Field(None, ge=0, le=1000)

    @field_validator("data_gb")
    @classmethod
    def round_gb(cls, v: float | None) -> float | None:
        return None if v is None else round(v, 2)  # custom logic; raise ValueError to reject
```

**Lax vs strict mode.** By default Pydantic is *lax*: `"300"` becomes `300`. That's convenient for HTTP
query strings and LLM output (models sometimes quote numbers). *Strict* mode (`ConfigDict(strict=True)`)
rejects `"300"` for an `int`. Use strict where a wrong type signals a real bug, such as internal
service-to-service contracts.

**Pydantic vs the alternatives** (a common interview question):

| | Validates at runtime? | Coerces types? | JSON Schema? | Use for |
|---|---|---|---|---|
| `dict` | No | No | No | Throwaway scripts |
| `TypedDict` | No (static checkers only) | No | No | Typing existing dicts cheaply |
| `@dataclass` | No | No | No | Internal data holders you construct yourself |
| **Pydantic `BaseModel`** | **Yes** | Yes (lax) | **Yes** | **Anything crossing a boundary** |

Rule of thumb: **Pydantic at the edges, plain classes inside** if performance matters.

**pydantic-settings** (already in `config.py`): `BaseSettings` reads fields from environment variables
and `.env`, matching names case-insensitively, with the same validation. So `PRICE_INPUT_PER_M=abc`
fails at startup with a clear error instead of producing a wrong cost silently a week later.
Failing fast on bad config is a production habit worth naming in interviews.

> ⚠️ **A gotcha you may hit in 00b:** a provider's structured outputs support only a *subset* of JSON
> Schema. Some keywords (e.g. some numeric constraints or formats) may be rejected or ignored by the
> API. Pydantic still validates on your side after parsing, so the constraint isn't lost. Check the
> OpenAI structured-outputs docs for the supported keywords if a call errors.

### 3. pytest: the fast feedback loop

```python
# tests/test_pricing.py: pytest finds files named test_*.py and functions named test_*
def test_no_cache():
    assert estimate_cost(3_000, 300, PRICES) == pytest.approx(0.00168)   # approx for floats!
```

The four features you'll use (all appear in the tests I wrote, so read them as examples):
- **Plain `assert`**: pytest rewrites it to show both sides on failure.
- **`@pytest.mark.parametrize`**: one test, many inputs (`test_rejects_invalid_counts`).
- **Fixtures**: reusable setup injected by argument name (`llm`, `tmp_path`, `monkeypatch` in `test_llm.py`).
  `tmp_path` gives a fresh temp dir. `monkeypatch` temporarily replaces an attribute or env var and undoes it after the test.
- **Markers**: we mark real-API tests `@pytest.mark.live` and skip them by default (see `addopts` in
  `pyproject.toml`). `uv run pytest -m live` runs them deliberately.

**How to test LLM code, the important part.** You can't unit-test "is the answer good?", because
that's what evals are for (Phase 3). You *can* unit-test everything around the model deterministically:
- **Fakes** replace the API client (`FakeResponses` in `test_llm.py`). They test your plumbing (cost math,
  logging, argument passing) in milliseconds, for free, offline, and in CI.
- **Live tests**: a few, marked, run on demand. They catch API contract changes.
- **Evals**: hundreds of cases, measuring *quality*. Different tool, different cadence.

That three-layer answer comes up in AI Engineer interviews a lot. Learn it.

### 4. ruff: linter + formatter

- A **linter** finds likely bugs and bad patterns *without running the code* (unused imports,
  undefined names, `except:` that swallows everything, mutable default arguments).
- A **formatter** rewrites code into one consistent style, so code review is about logic, not spaces.

Ruff does both, written in Rust and roughly 10–100× faster than what it replaces: **flake8** (lint),
**black** (format), **isort** (import order), **pyupgrade** (modern syntax). One tool, one config section.

Our config (`pyproject.toml → [tool.ruff.lint] select`):

| Code | Rule set | Catches |
|---|---|---|
| `E` | pycodestyle | Style errors |
| `F` | pyflakes | Unused imports/variables, undefined names (real bugs) |
| `I` | isort | Import ordering |
| `UP` | pyupgrade | Old syntax, e.g. `Optional[X]` → `X \| None` |
| `B` | flake8-bugbear | Likely bugs, e.g. `def f(x=[])` (shared mutable default) |

```bash
uv run ruff check .          # report problems
uv run ruff check . --fix    # auto-fix the safe ones
uv run ruff format .         # reformat everything
```

### 5. pre-commit: quality gates that run before every commit

Git can run a script before each commit (a *git hook*). **pre-commit** is a framework that manages
those hooks from one file, `.pre-commit-config.yaml`. Each hook is pinned to a version, installed in
its own isolated environment, and runs **only on the files you're committing**, so it stays fast.

Ours runs: ruff (lint + fix), ruff-format, whitespace/EOF fixers, YAML/TOML syntax checks, a large-file
guard, a private-key detector, **a block on committing `.env`**, and **a regex that rejects anything
shaped like an OpenAI key**. The last two matter: leaked API keys are among the most common real
incidents, and bots scrape GitHub for them within minutes.

If a hook fails, the commit is aborted. If a hook *fixed* files (formatting), stage them and commit again.

**pre-commit vs CI:** pre-commit is fast local feedback, but it's optional (`git commit --no-verify`
skips it). **CI is the enforced gate.** We'll run the same checks plus tests and evals in CI in Phase 3.
You want both: catch problems in seconds locally, and guarantee them on the server.

---

## Part C — Build (~1 h)

1. **Pydantic exercise:** in `src/telco_support_agent/schemas.py`, follow the `TODO(you)` to make
   `tests/test_schemas.py` pass: reject negative values, set sane upper bounds, forbid unknown fields.
   Run `uv run pytest tests/test_schemas.py -v`.
2. **Play in a REPL** (`uv run python`): build `UsageProfile` objects with good and bad data, read the
   error messages, and print `UsageProfile.model_json_schema()`. Find your descriptions in it.
3. **Update pinned hook versions:** `uv run pre-commit autoupdate` (I pinned older known versions).
   Then `uv run pre-commit run --all-files` and see what it fixes.
4. **Watch the gates work.** Create `scratch.py` containing `import os` (unused) and the line
   `key = "sk-abcdefghijklmnopqrstuvwxyz123456"`, then `git add scratch.py && git commit -m test`.
   Both problems should block the commit. Delete the file afterwards.
5. Try `git add .env` → commit. It should be refused. (That's also why `.env` is in `.gitignore`.)

Then move on to [Lesson 00b](00b-llm-api-fundamentals.md). Say "review lesson 00a" whenever you want
the drill and a code review. You can do 00a and 00b together.

---

## Part D — Interview drill

These are the questions you should be ready for. When you say "review lesson 00a", I quiz you in **short format** (multiple choice, one-liners, keywords), then show the full [model answers in interview-bank §00a](../interview-bank.md#00a--python-engineering-toolkit) to read or say out loud.

1. Why is Pydantic so common in LLM applications specifically?
2. Pydantic `BaseModel` vs `@dataclass` vs `TypedDict`: when would you use each?
3. Lax vs strict validation: what's the difference and when would you choose strict?
4. How do you test code that calls an LLM?
5. What's the difference between a linter and a formatter? Why has ruff replaced flake8/black/isort in many teams?
6. If you have pre-commit hooks, why do you also need CI checks?
7. What is a pytest fixture? Give an example.
8. How do you keep secrets out of a repo, and what do you do if a key leaks anyway?
