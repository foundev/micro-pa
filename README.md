# micro-pa

[![ci](https://github.com/foundev/micro-pa/actions/workflows/ci.yml/badge.svg)](https://github.com/foundev/micro-pa/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/foundev/micro-pa)](https://github.com/foundev/micro-pa/releases)
[![go](https://img.shields.io/badge/go-1.25-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![deps](https://img.shields.io/badge/dependencies-none-brightgreen)](#development)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A small personal assistant in your terminal: one Go binary, no services, no
database. It talks to a model, remembers things between sessions, keeps a task
list, and can read files or run shell commands when you let it.

This is a deliberately minimal take on the "personal agent" idea. The whole
thing is a few hundred lines of standard library Go, and the state is plain JSON
you can read and edit yourself.

## Quick start

Go 1.25 or newer. There are no third-party dependencies, so nothing is
downloaded to build it.

```sh
go build -o mpa ./cmd/mpa

export OPENROUTER_API_KEY=sk-or-...
./mpa "what should I focus on today?"
```

Prebuilt binaries for Linux, macOS and Windows are on the
[releases page](https://github.com/foundev/micro-pa/releases). Each archive
unpacks to a single `mpa` binary, plus the README and license. Release builds
report the tagged version from `mpa -version`; a local `go build` reports `dev`.

Other ways to start it:

```sh
./mpa                            # interactive session
./mpa -once "summarize my notes" # answer one question, print, exit
echo "long question" | ./mpa -once -
./mpa -provider mock             # offline, canned replies, no key needed
```

The first run with no key prints a short explanation and stops. `./mpa -init`
writes a starter `config.json` you can edit.

## What it does

- **Chat** with any OpenAI-compatible `/chat/completions` endpoint: OpenRouter
  (the default), a local vLLM or llama.cpp server, or anything else with the
  same shape. The default model is the OpenRouter alias
  `~deepseek/deepseek-flash-latest`, which tracks the current DeepSeek Flash
  release. Pin an exact slug in `config.json` if you want a frozen version.
- **Remembers** durable notes. Saved notes and open tasks are folded into the
  system prompt each turn, so it picks up where you left off in a new process,
  and it can search the older ones with the `recall` tool.
- **Tracks tasks** with a small todo list it can add to and close out.
- **Reads files** you point it at.
- **Runs shell commands** only when you enable shell mode, and it asks before
  each command when you are in the interactive session.
- **Saves conversations** as sessions you can list and switch between.

## Commands

Anything that is not a slash command goes to the model.

| Command | What it does |
| --- | --- |
| `/help` | List the commands |
| `/tools` | Show the tools the model can call |
| `/memory [query]` | Show saved notes, optionally filtered |
| `/remember <text>` | Save a note yourself, without the model |
| `/forget <id>` | Delete a note |
| `/tasks [all]` | Show open tasks, or all of them |
| `/done <id>` | Complete a task |
| `/new` | Start a fresh conversation |
| `/sessions`, `/session <name>` | List or switch conversations |
| `/model [slug]` | Show or change the model |
| `/shell on\|off` | Enable the shell tool |
| `/config` | Show the active settings, with the key masked |
| `/quit` | Leave |

## Flags

| Flag | Meaning |
| --- | --- |
| `-once <prompt>` | Answer one question and exit. `-` reads the prompt from stdin. |
| `-provider` | `openai` (any compatible API) or `mock` (offline). |
| `-model` | Model slug. |
| `-base-url` | API base URL. |
| `-api-key` | API key. Prefer the environment variable. |
| `-home` | State directory, default `~/.micro-pa`. |
| `-session` | Conversation to continue, default `default`. |
| `-persona` | Extra instructions about you, added to the system prompt. |
| `-allow-shell` | Let the model run shell commands. |
| `-init` | Write a starter `config.json` and exit. |
| `-tools`, `-sessions` | Print the tool or session list and exit. |
| `-v` | Print the system prompt at startup. |
| `-version` | Print the version. |

## Configuration

Settings come from, in increasing priority: built-in defaults, then
`~/.micro-pa/config.json`, then environment variables, then flags.

```json
{
  "provider": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "",
  "model": "~deepseek/deepseek-flash-latest",
  "home": "/home/you/.micro-pa",
  "persona": "Keep answers short. I mostly work in Go, on Linux.",
  "allow_shell": false,
  "max_turns": 8,
  "timeout_seconds": 120,
  "temperature": 0.7
}
```

Keeping `api_key` out of the file and in the environment is the better habit;
the flag and the file are there for convenience.

| Variable | Purpose |
| --- | --- |
| `MPA_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY` | API key, first one set wins |
| `MPA_MODEL`, `MPA_BASE_URL`, `MPA_PROVIDER` | Backend selection |
| `MPA_HOME` | State directory |
| `MPA_PERSONA` | Extra system instructions |
| `MPA_ALLOW_SHELL` | `1` or `true` to enable the shell tool |
| `MPA_MAX_TURNS`, `MPA_TIMEOUT_SECONDS`, `MPA_TEMPERATURE` | Limits |

Pointing at a local server:

```sh
MPA_BASE_URL=http://localhost:8000/v1 MPA_API_KEY=dummy MPA_MODEL=local ./mpa
```

## How a turn works

1. The system prompt is rebuilt, with the tool schemas and a summary of your
   notes and open tasks.
2. The conversation is sent to the model.
3. If the reply contains a `tool_call` block, the tool runs and its result goes
   back to the model as a `tool` message. Tool activity is printed to stderr, so
   piped stdout stays clean.
4. Repeat until the model answers without calling a tool, or `max_turns` is hit.

Thinking traces (`<think>`, `<reasoning>`) are stripped before the reply is
stored, so they do not pile up in the context window.

## Tools

`now`, `remember`, `recall`, `forget`, `task_add`, `task_list`, `task_done`,
`read_file`, and `shell` (off by default).

## Files

Everything lives under `~/.micro-pa`:

```
config.json            settings, 0600
memory.json            saved notes
tasks.json             todo list
sessions/<name>.json   saved conversations
```

Writes are atomic: the file is replaced by renaming a temporary file into place,
so an interrupted write cannot corrupt your notes.

## A note on the shell tool

With `-allow-shell`, the model can run any command you can run. It only sees
what a command prints, but that is still a real capability, and text from a file
or a command can try to talk the model into running something else. In the
interactive session every command is shown and needs a `y`. In `-once` mode
there is nobody to ask, so commands run unattended; only enable it there if you
trust what you are feeding in. Shell access is off unless you ask for it.

## Development

```sh
go test ./...        # unit tests, no network
go test -race ./...
go vet ./...
```

The layout is small on purpose:

```
cmd/mpa            the CLI: flags, REPL, slash commands
internal/agent     the tool-calling loop and the system prompt
internal/llm       the OpenAI-compatible client, plus an offline mock
internal/tools     the tools, their schemas, and their safety checks
internal/store     JSON persistence for notes, tasks and sessions
internal/config    config file, environment and defaults
```

Tests never touch the network. The HTTP client takes a swappable transport, so
the CLI tests drive the full request path in-process.

Releases are cut from tags: pushing a `v*` tag runs
`.github/workflows/release.yml`, which cross-compiles Linux, macOS and Windows
binaries (amd64 and arm64), stamps the tag into `mpa -version`, and publishes
the archives with checksums to the releases page.

## Not here yet

The things a bigger personal agent has that this one deliberately leaves out:
streaming replies, a Telegram or Slack front end, scheduled runs, subagents,
web search, a sandbox for the shell, MCP, and per-conversation model settings.

## License

MIT. See [LICENSE](LICENSE).
