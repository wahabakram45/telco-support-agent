# Lesson 04b — Docker: packaging the API so it runs the same everywhere

> **Goal:** understand what containers are and how images are built, well enough to write a
> production-style Dockerfile and explain every line in an interview: images vs containers, layers
> and caching, multi-stage builds, ports, env vars and secrets, volumes, `.dockerignore`, non-root
> users and health checks. Containerize our FastAPI service and run it locally.
> **Time:** ~3 hours (1.5 h reading, 30 min install, 30 min building, 30 min exploring + quiz).

Right now the API runs on your Mac because you set it up by hand: Python 3.12 through uv, ~70
packages, the Chroma index you built, a `.env` file. A teammate, a CI runner or a cloud server would
need to repeat every step exactly, on a different operating system. Docker packages all of it into
one artifact that runs the same anywhere.

---

## Part A — Concepts

### 1. The problem Docker solves

"It works on my machine" usually means one of these differs between machines:

| What differs | Our example |
|---|---|
| Language version | Python 3.12 on your Mac, 3.10 on the server |
| Packages | `chromadb 1.5.9` here, whatever `pip install` picked there |
| System libraries | Linux servers vs macOS; some wheels need specific C libraries |
| CPU architecture | Your Mac is **arm64** (Apple Silicon); most cloud servers are **amd64** (x86) |
| Files and setup | The index has to exist at `data/chroma`; `logs/` must be writable |
| Config | Env vars like `OPENAI_API_KEY`, `LLM_MODEL` |

A **container image** bundles the OS userland, Python, packages, our code and the index into one
read-only artifact with an ID. Whatever runs it, a laptop, CI or Kubernetes, runs the exact same
bytes. Only config and secrets come from outside, at run time.

### 2. Containers vs virtual machines

| | Virtual machine | Container |
|---|---|---|
| What it virtualizes | Hardware: each VM runs a full OS with its own kernel | The OS: containers share the host's Linux kernel |
| Start time | Tens of seconds | Under a second (it's just a process) |
| Size | GBs | MBs to a few hundred MB |
| Isolation | Strong (separate kernels) | Good, but weaker (shared kernel) |

A container is **just a normal Linux process** with two kernel features applied:
- **namespaces** decide what it can *see*: its own filesystem, process list (it thinks it's PID 1),
  network interfaces and hostname;
- **cgroups** limit what it can *use*: CPU, memory.

**Why you need Colima or Docker Desktop on a Mac:** containers need a *Linux* kernel, and macOS
isn't Linux. So on a Mac, Docker quietly runs a small Linux VM and your containers run inside it.
On a Linux server there's no VM: containers run directly on the host.

### 3. Image, container, registry

- **Image:** a read-only template: a stack of filesystem **layers** plus metadata (which command
  to run, env vars, the exposed port). Like a **class**.
- **Container:** a running (or stopped) instance of an image, with a thin **writable layer** on
  top. Like an **object**. You can start 10 containers from one image. When a container is
  deleted, its writable layer, and everything written to it, is gone.
- **Registry:** where images are stored and shared: Docker Hub, GitHub Container Registry (GHCR),
  AWS ECR, Google Artifact Registry. `docker push` / `docker pull`.
- **Tag vs digest:** `python:3.12-slim` is a **tag**: a movable label that the maintainers re-point
  when they publish security fixes. `python@sha256:4f1…` is a **digest**: the exact bytes, forever.
  Tags are readable; digests are reproducible. Production pipelines often pin digests.

```
Dockerfile ──docker build──► Image ──docker run──► Container(s)
                               │
                          docker push
                               ▼
                           Registry ──docker pull──► another machine
```

### 4. The Dockerfile, line by line

A Dockerfile is a recipe: each instruction runs in order, and most of them add a layer.

| Instruction | What it does | In our Dockerfile |
|---|---|---|
| `FROM` | The base image to start from. Starts a new **stage** | `python:3.12-slim` |
| `WORKDIR` | `cd` into a directory (creating it if needed) for all later steps | `/app` |
| `COPY` | Copy files from the **build context** (your folder) or another stage/image into the image | the lockfile, `src/`, the index |
| `RUN` | Run a command at **build** time, saving the result as a layer | `uv sync`, `useradd` |
| `ENV` | Set an env var for later steps *and* for the running container | `PATH`, `PYTHONUNBUFFERED` |
| `USER` | Which user later steps and the container run as | `app` |
| `EXPOSE` | **Documents** the port the app listens on. Doesn't publish anything | `8000` |
| `HEALTHCHECK` | A command Docker runs periodically to mark the container healthy/unhealthy | calls `/health` |
| `CMD` | The default command when the container **starts** | `uvicorn …` |

Key distinctions interviewers like:
- **`RUN` vs `CMD`:** `RUN` happens once, at *build* time (install packages). `CMD` happens every
  time a container *starts* (start the server).
- **`CMD` vs `ENTRYPOINT`:** `ENTRYPOINT` is the fixed executable; `CMD` gives default arguments
  that `docker run image <args>` replaces. With only `CMD` (our case), `docker run image bash` runs
  bash instead of the server, which is handy for debugging.
- **Exec form vs shell form:**
  ```dockerfile
  CMD ["uvicorn", "telco_support_agent.api:app"]   # exec form: uvicorn IS the main process (PID 1)
  CMD uvicorn telco_support_agent.api:app          # shell form: runs /bin/sh -c "uvicorn …"
  ```
  `docker stop` sends **SIGTERM** to PID 1. In exec form, uvicorn receives it and shuts down
  gracefully (finishes in-flight requests). In shell form, `sh` gets it and usually doesn't pass it
  on, so after 10 seconds Docker kills everything with SIGKILL: requests cut off mid-answer.
  **Always use exec form for `CMD`.**
- **`COPY` vs `ADD`:** `ADD` can also download URLs and unpack archives, which is surprising
  behaviour. Prefer `COPY`.
- **`ENV` vs `ARG`:** `ARG` exists only during the build; `ENV` persists into the container. Both
  are visible in the image history, so **neither is for secrets** (§9).

### 5. Layers and caching: why the order of lines matters

Each `RUN`, `COPY` and `ADD` creates a **layer**: a snapshot of the filesystem changes that step
made. Layers are cached. On the next build, Docker reuses each cached layer **until the first step
whose inputs changed**. From that step on, *everything* is rebuilt.

- For `COPY`, the "input" is the content of the copied files.
- For `RUN`, it's the command text (plus all the layers before it).

So **put things that change rarely first, and things that change often last.**

**❌ Naive order:**
```dockerfile
COPY . .                       # any code edit changes this layer…
RUN uv sync --locked           # …so all ~70 packages reinstall on EVERY build (a minute or more)
```
**✅ Cache-friendly order (yours, Task 1):**
```dockerfile
COPY pyproject.toml uv.lock ./                    # changes only when dependencies change
RUN uv sync --locked --no-dev --no-install-project   # cached: dependencies only (the slow part)
COPY src ./src                                    # changes on every code edit
RUN uv sync --locked --no-dev --no-editable       # fast: installs just our small package
```
Edit `api.py` and rebuild: the first two steps say `CACHED`, and the build takes seconds.

This is the same trick in every language: Node copies `package.json` + `package-lock.json` and runs
`npm ci` before copying the code; Python copies `requirements.txt` before the app.

The uv flags, since you'll be asked:
- `--locked`: fail if `uv.lock` is out of date with `pyproject.toml`, instead of silently
  re-resolving. The image gets *exactly* the versions you tested with.
- `--no-dev`: skip pytest, ruff and pre-commit; they don't belong in production.
- `--no-install-project`: install the dependencies but not our package yet (its code isn't copied).
- `--no-editable`: install our package as a normal copy into `site-packages`, not a link to
  `src/`. That's what lets the runtime stage drop `src/` entirely.

### 6. Choosing a base image

| Base | Size | Notes |
|---|---|---|
| `python:3.12` | ~1 GB | Full Debian with compilers. Convenient, big, more to patch |
| **`python:3.12-slim`** | ~120 MB | Minimal Debian. Uses normal Linux wheels (numpy, onnxruntime, chromadb's Rust bindings install prebuilt). **Our choice** |
| `python:3.12-alpine` | ~50 MB | Uses **musl** instead of glibc, so many prebuilt wheels don't fit and packages compile from source: slow builds, sometimes failures, sometimes slower code. A classic trap for ML images |
| Distroless / Chainguard | Small | No shell, no package manager: fewer vulnerabilities, but harder to debug |
| `nvidia/cuda` | GBs | Only when you need a GPU (local models, Phase 8 maybe) |

Pin the version (`3.12-slim`, never `latest`), so a rebuild next month doesn't silently move to
Python 3.14.

### 7. Multi-stage builds

A Dockerfile can have several `FROM`s. Each starts a fresh **stage**, and a later stage can copy
files out of an earlier one. Only the **last** stage becomes the image.

```
Stage 1 "builder"                          Stage 2 (the final image)
 python:3.12-slim                           python:3.12-slim
 + uv binary                                + a non-root user
 + pyproject.toml, uv.lock                  + /app/.venv   ◄── copied from builder
 + src/                                     + data/chroma
 + /app/.venv (everything installed)        + CMD uvicorn …
 ✗ thrown away                              ✓ shipped
```
Why: the builder needs tools the server never uses: uv, source code, sometimes compilers. Leaving
them out makes the image **smaller** (faster to pull and start) and **safer** (less software means
fewer vulnerabilities to patch, and fewer tools for an attacker who gets in).

**Honest numbers for us:** the virtualenv alone is ~480 MB, so expect an image of roughly
600 MB. Most of it is `chromadb`'s own dependencies: `onnxruntime` (75 MB), the `kubernetes` client
(74 MB), Chroma's Rust bindings (50 MB), `grpc` (41 MB). We use none of them except the bindings,
because we pass our own embeddings. In production you'd run Chroma (or Qdrant, pgvector…) as a
**separate service**, and the API would install only a thin client (`chromadb-client`). That alone
would cut the image a lot. Noted for later.

### 8. Networking and ports

A container gets its **own network namespace**, with its own `localhost`.

- **Inside the container, `127.0.0.1` means the container itself.** A server bound to
  `127.0.0.1:8000` accepts connections only from inside the container, so your Mac can't reach it.
  That's why our `CMD` binds **`0.0.0.0`** (all interfaces). This is the #1 "my container runs but
  I can't connect" bug.
- **`-p 8000:8000`** *publishes* a port: `host_port:container_port`. `-p 9000:8000` would make the
  API reachable on your Mac at `localhost:9000`.
- **`EXPOSE 8000`** publishes nothing; it's documentation (and a default for some tools).
- Containers on the same Docker network reach each other **by service name**: in Compose, the API
  could call `http://chroma:8000`. That's how the API would talk to a separate vector DB later.

### 9. Configuration and secrets

The rule (from the **12-factor app** principles): **the same image runs everywhere; only
config changes, and config comes from the environment.** One image is promoted from dev to staging
to production with different env vars, never rebuilt per environment.

Our `Settings` class already reads env vars (that's what pydantic-settings does), so at run time:
```bash
docker run --env-file .env telco-support-agent      # reads KEY=VALUE lines into the container's env
docker run -e LLM_MODEL=gpt-5.4-mini …              # or one at a time
```

**Never put secrets in the image.** Anyone who can pull the image can read every layer:
- ❌ `COPY .env .`: the key is a file inside a layer forever. Deleting it in a later step doesn't
  help; the earlier layer still contains it.
- ❌ `ENV OPENAI_API_KEY=sk-…` or `ARG OPENAI_API_KEY`: visible with `docker history` and
  `docker inspect`.
- ✅ Inject at **run** time: `--env-file`, `-e`, or in production a **secret manager** (AWS Secrets
  Manager, GCP Secret Manager, Vault, Kubernetes Secrets) that provides env vars or mounted files.
- ✅ If a *build* step genuinely needs a secret (a private package index), use BuildKit's
  `RUN --mount=type=secret,…`, which is available during that step and never saved in a layer.

Two layers of protection in our repo: `.dockerignore` keeps `.env` out of the build entirely, and
`tests/test_docker.py` fails if the Dockerfile ever copies `.env` or sets a key.

> **Gotcha:** `--env-file` does **not** strip quotes. A line `OPENAI_API_KEY="sk-abc"` gives the
> container the value `"sk-abc"` *with* the quotes, and OpenAI returns 401. Inline `# comments`
> after a value aren't supported either. (Compose's `env_file:` does handle quotes.)

### 10. Volumes: where state lives

A container's writable layer **disappears when the container is removed**. Our app writes three
things at run time: `logs/*.jsonl`, the embedding cache in `.cache/`, and Chroma's SQLite file. How
to keep what matters:

| Mechanism | Syntax | Use for |
|---|---|---|
| **Bind mount** | `-v ./logs:/app/logs` | Sharing a host folder: development, seeing logs on your Mac |
| **Named volume** | `-v telco-logs:/app/logs` | Data managed by Docker that outlives containers (databases) |
| **tmpfs** | `--tmpfs /app/.cache` | Scratch space in memory, never written to disk |

And the bigger decision, **where does the index live?**

| Option | Pros | Cons |
|---|---|---|
| **Bake into the image** (`COPY data/chroma`) — **our choice** | Code and index are versioned together: rolling back the image rolls back the index; the image is self-contained; matches our model-mismatch check | Rebuild the image to update the index; image grows with the index |
| Mount at run time (`-v ./data/chroma:/app/data/chroma`) | Update the index without rebuilding | The image alone doesn't work; easy to pair the wrong index with the code |
| Separate vector DB service (Chroma server, Qdrant, pgvector) | Scales, shared by many API copies, updated live | One more service to run: the production answer for large or frequently updated indexes |

Our index is 5 MB and changes only when documents change, so baking it in is the simple, safe
choice now.

**Logs:** in production you don't write log files inside containers at all. You log to
**stdout/stderr**, and the platform (Docker, Kubernetes, CloudWatch, Datadog) collects them. That's
why `docker logs` works, and why we set `PYTHONUNBUFFERED=1` (otherwise Python buffers output and
logs appear late or are lost on a crash). Our JSONL files are a learning aid; the bind mount keeps
them while we're developing.

### 11. The build context and `.dockerignore`

`docker build .` first sends the **build context** (the folder, here `.`) to the Docker engine.
Without a `.dockerignore`, that includes `.venv` (430 MB), `.git`, `logs`, caches… and `.env`.
Then a careless `COPY . .` puts all of it in the image.

Two styles:
- **Blocklist:** list what to exclude (`.env`, `.venv`, `.git`…). Easy to forget a new file.
- **Allowlist (ours):** exclude everything (`*`), then re-include only what the image needs with
  `!`. A new secret or big folder is excluded by default.

### 12. Running containers well

- **Non-root user.** By default, processes in a container run as root. If an attacker exploits the
  app, root inside the container makes escaping or damaging things much easier. We create `app`
  (UID 1000) and switch to it with `USER app`. Many platforms (OpenShift, hardened Kubernetes)
  refuse root containers outright.
- **One process per container.** Run the API in one container and the vector DB in another, not
  both in one. Each can be scaled, restarted and updated independently.
- **Scaling: more containers, not more workers.** In Lesson 04a we said `uvicorn --workers 4` uses
  4 cores. In a container platform you usually run **one worker per container** and let the
  orchestrator run more containers. It can then restart, place and scale each one separately.
- **Health checks.** Our `HEALTHCHECK` calls `/health` every 30 s; `docker ps` shows
  `(healthy)` or `(unhealthy)`. It uses Python, because the slim image has no `curl`. (Kubernetes
  ignores Docker's `HEALTHCHECK` and uses its own liveness and readiness probes, which point at
  the same endpoint.)
- **CPU architecture.** Built on your Apple Silicon Mac, the image is **linux/arm64**. An x86
  cloud server needs **linux/amd64**: `docker build --platform linux/amd64 …` (slower on a Mac,
  through emulation), or `docker buildx build --platform linux/amd64,linux/arm64` for both. CI
  usually builds on amd64 runners, so this rarely bites in practice, but "exec format error" is
  what it looks like when it does.

### 13. Docker Compose

Typing `docker run --rm -p 8000:8000 --env-file .env -v ./logs:/app/logs telco-support-agent`
every time gets old. `compose.yaml` writes those flags down once, and `docker compose up --build`
builds and runs. Its real strength is **several services**: later, `api` + `chroma` (or `qdrant`)
+ `redis` for caching, on a shared network, all started together. It's for local development and
simple deployments; production usually means Kubernetes, ECS or Cloud Run, which read the same
image.

### 14. Our decisions

- `python:3.12-slim` base, pinned; uv copied from its official image, also pinned (`0.13.6`).
- Multi-stage: builder installs with `uv sync --locked --no-dev`; runtime copies only `.venv`.
- Dependencies before code, so code edits rebuild in seconds.
- The index baked into the image; logs and cache written to folders owned by the app user.
- Non-root `app` user, exec-form `CMD` on `0.0.0.0:8000`, `HEALTHCHECK` on `/health`.
- Secrets only at run time (`--env-file`); `.dockerignore` as an allowlist; static tests enforce both.

---

## Part B — Build

I wrote: `.dockerignore`, `compose.yaml`, most of the `Dockerfile`, and `tests/test_docker.py`
(static checks that run without Docker). I also ran the exact `uv` steps and the server command
from the Dockerfile outside Docker to check them: `/health` and `/ask` both answered.
**You write the two parts that matter most.**

### Task 0 — Install Docker (~30 min, mostly downloading)

Docker on a Mac needs a Linux VM. Options:
- **Colima** (recommended here): open source (MIT), command line only, installs with Homebrew. No
  licensing questions.
- **Docker Desktop:** the official app with a GUI. Free for personal use, but larger companies need
  a paid subscription, which matters on a work machine.
- **OrbStack:** fast and polished; free for personal use only.

```bash
brew install colima docker docker-buildx docker-compose
# Tell the docker CLI where Homebrew put the buildx and compose plugins:
mkdir -p ~/.docker/cli-plugins
ln -sfn "$(brew --prefix)/opt/docker-buildx/bin/docker-buildx" ~/.docker/cli-plugins/docker-buildx
ln -sfn "$(brew --prefix)/opt/docker-compose/bin/docker-compose" ~/.docker/cli-plugins/docker-compose
colima start --cpu 2 --memory 4      # starts the Linux VM (run again after a reboot)
docker run --rm hello-world          # should print "Hello from Docker!"
```

### Task 1 — Install dependencies in a cache-friendly order (~15 min)
In the builder stage of `Dockerfile`, replace the Task 1 TODO with **four lines** (two `COPY`, two
`RUN`), following §5. The TODO comment lists the exact `uv` commands.

### Task 2 — Run safely and reachably (~10 min)
At the bottom of `Dockerfile`: switch to the `app` user, then a `CMD` in **exec form** that starts
`uvicorn` serving `telco_support_agent.api:app` on host `0.0.0.0`, port `8000`. Two lines.

Check: `uv run pytest tests/test_docker.py -v` (14 tests; 4 wait for you).

### Task 3 — Build, run and explore (~30 min)
The index isn't in git, so make sure it exists (`ls data/chroma`; if not, `uv run python scripts/build_index.py`).

```bash
docker build -t telco-support-agent .                   # first build: a few minutes
docker image ls telco-support-agent                     # how big is it?
docker run --rm -d --name telco -p 8000:8000 --env-file .env telco-support-agent
docker ps                                               # wait ~30 s for "(healthy)"
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "How much is the GCC day pass?", "market": "UAE"}'
```
Then explore:
1. `docker logs -f telco`: the request log lines, from stdout. (Ctrl+C stops following, not the container.)
2. `docker exec -it telco whoami` → `app`, not root. `docker exec -it telco ls -la /app`: no
   `src/`, no `.env`, no `tests/`.
3. `docker history telco-support-agent`: one row per layer, with sizes. Find the big one.
4. **The caching experiment:** change a word in a docstring in `api.py`, run `docker build` again,
   and watch the steps say `CACHED` until `COPY src`. Note how long it took compared with the first build.
5. `docker stop telco`: it stops within a second or two (exec form, so uvicorn got SIGTERM).
6. Compose: `docker compose up --build`, query it again, then look in `./logs/` on your Mac: the
   bind mount at work. Ctrl+C, then `docker compose down`.

If `/ask` returns 401, check `.env` for quotes around values (§9 gotcha).

When you're done, commit and say "lesson 04b done". I'll clean up, then quiz you.

### Command cheat sheet

| Command | What it does |
|---|---|
| `docker build -t name .` | Build an image from `./Dockerfile`, tag it `name` |
| `docker run --rm -p H:C --env-file .env name` | Start a container; remove it when it stops |
| `docker run -d --name x …` | Run in the background (detached), named `x` |
| `docker ps` / `docker ps -a` | Running containers / all containers |
| `docker logs -f x` | Follow a container's output |
| `docker exec -it x bash` | Open a shell inside a running container (slim has `bash`) |
| `docker stop x` / `docker rm x` | Stop (SIGTERM, then SIGKILL after 10 s) / delete |
| `docker image ls` / `docker history name` | List images / show an image's layers |
| `docker system df` / `docker system prune` | Disk used by Docker / clean up unused data |
| `docker compose up --build` / `down` | Build and start everything in `compose.yaml` / stop and remove |

---

## Part C — Interview drill

Short-format quiz on "review lesson 04b". Full model answers:
[interview-bank §04b](../interview-bank.md#04b--docker).

1. What's the difference between an image and a container? Between a container and a VM?
2. Why does the order of instructions in a Dockerfile matter? Give the dependency example.
3. What is a multi-stage build, and why use one?
4. How do you give a container secrets, and what must you never do?
5. The container is running but `curl localhost:8000` fails. What do you check?
6. Why run as a non-root user, and why exec-form `CMD`?
7. Which base image would you pick for a Python ML service, and why not Alpine?
8. Where should state live (logs, the index, uploads) when containers are disposable?
