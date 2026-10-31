# Working with pi on the local model

[pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) talks to the server that `bin/strata-serve` runs through
an ordinary provider entry in its own `~/.pi/agent/models.json`; the repository ships no pi profile.

## One-time setup

Add the provider to `~/.pi/agent/models.json` (create the file if it is missing, merge the `strata` entry into `providers`
if it exists):

```json
{
  "providers": {
    "strata": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "strata-local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        {
          "id": "qwen3.6-35b-a3b",
          "name": "Qwen3.6 35B-A3B (local)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 8192,
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        }
      ]
    }
  }
}
```

`pi --list-models` then shows `strata qwen3.6-35b-a3b`. To make it the model pi starts with, set it in `~/.pi/agent/settings.json`; the
`modelThinkingLevels` entry turns thinking off for this model only, so other models keep their own level:

```json
{
  "defaultProvider": "strata",
  "defaultModel": "qwen3.6-35b-a3b",
  "modelThinkingLevels": { "strata/qwen3.6-35b-a3b": "off" }
}
```

pi needs the server to be up: with `bin/strata-serve` stopped, its first request fails with a connection error. Switch to
another model with `/model` or `pi --model PROVIDER/ID`.

```sh
bin/strata-serve                                                  # terminal 1: the model on http://127.0.0.1:8080
cd ~/some/project && pi                                           # terminal 2: interactive (the local model is the default)
pi -p "list the files here and count the lines of b.txt"           # one shot
pi --model strata/qwen3.6-35b-a3b:off                              # the same model chosen explicitly, thinking off
```

## What the provider entry sets

| Setting | Why |
| --- | --- |
| API `openai-completions`, thinking format `qwen-chat-template` | the server speaks the OpenAI chat API and switches thinking through the chat template |
| `contextWindow` 65536, `maxTokens` 8192 | the engine's window; pi trims and compacts against it. Keep it equal to the server's `--context` |
| `modelThinkingLevels` `off` (or the `:off` suffix of `--model`) | thinking costs tokens that decode at 25-35 tok/s; turn it on per task |

Thinking: `pi --model strata/qwen3.6-35b-a3b:high` (levels `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`). pi's
compaction defaults (`compaction.reserveTokens`, `keepRecentTokens` in `settings.json`) are sized for 200K windows; with
the 64K window 8192 and 12000 work well, set them in `settings.json` if you use the model a lot.

## Running it safely

pi has no sandbox. It runs the model's commands with the rights of your account and asks nothing before each one (pi's own
guide, "Run Pi safely", says the same: a working folder does not stop a command from reaching other paths). A small local
model can be wrong, and text in a file it reads can steer it. `sudo rm -rf /` cannot run unattended where `sudo` asks for a
password, but nothing needs `sudo` to run `rm -rf ~/*`, to read `~/.ssh` and the tokens in `~/.pi/agent/auth.json`, or to
push with your keys.

`tools/spi` starts pi inside a bubblewrap sandbox, so a wrong command can only reach what is mounted:

| The model's commands see | How |
| --- | --- |
| the folder you start it in, and everything below it | read-write |
| `/usr`, a few `/etc` files, the node installation | read-only |
| an empty home: no `~/.ssh`, no other projects, no tokens, no `auth.json` of your own pi | a temporary home directory |
| everything above the folder, `/` and the home included: only the empty chain of directories down to it, read-only, so nothing can be read, created or changed there | `--remount-ro` on `/` and on the home |
| `.git/hooks` and `.git/config` of the folder: read-only | a hook or a config written by the model would run later on the host, outside the sandbox |
| no controlling terminal: the model's commands cannot type into your shell | `--new-session`; because of it the kernel sends no SIGWINCH, so `spi` polls the window size and forwards each change to pi by hand |
| an empty environment: no `SSH_AUTH_SOCK`, no tokens in variables | `--clearenv` |
| no `sudo` (the "no new privileges" flag is set) and no capabilities | `--cap-drop ALL` |
| no network, except a bridge to the Strata server on `127.0.0.1:8080` | `--unshare-net` plus `socat` over a socket |
| `gh`, read-only, for the repositories of the folder: the token and the network stay outside | a broker on the host behind a second socket ([below](#gh-in-the-sandbox)) |
| its own profile, `models.json` and `settings.json` read-only, sessions kept | `~/.pi/agent-local` |

It refuses to start in `/`, in your home directory itself or in a hidden folder of it, and checks that the server answers.
Checked here: from the sandbox the home canary, a sibling folder, `~/.ssh` and `auth.json` do not exist, `touch /usr/bin/x`,
`rm /usr/bin/true` and `echo x > /etc/passwd` fail with "Read-only file system", `sudo` refuses, `curl https://example.com`
cannot resolve a name, and `rm -rf --no-preserve-root /` removes nothing outside the mounted folder (the host's `/usr/bin`,
canary files and secrets were intact afterwards).

### gh in the sandbox

`gh` inside the sandbox has no token and no network, so by itself it only says "not logged in". `spi` starts a small broker on
the host (`tools/spi_gh_broker.py`) and puts a `gh` in front of the real one that sends each command to it over a unix socket.
The broker checks the command against a fixed list, runs the real `gh` with your token and returns the output. The token never
enters the sandbox and the network stays closed.

| | |
| --- | --- |
| allowed | `pr view`, `list`, `diff`, `checks`, `status`; `issue view`, `list`; `run list`, `view`; `repo view`; `api` with GET under `repos/OWNER/REPO/` (pulls, issues, commits, compare, branches, check-runs, contents, git/trees, actions runs and jobs) |
| refused | everything that writes (create, comment, review, merge, close, edit, delete, `workflow run`), `auth`, `config`, `alias`, `extension`, `search`, `graphql`, `--jq`, `--template`, `--web`, `--input`, a field value that starts with `@` (gh would read that file on the host), headers other than `Accept`, and `--` |
| repositories | the `github.com` remotes in the folder's `.git/config` (read-only inside the sandbox), plus `SPI_GH_REPOS=owner/repo,owner/other`; `-R` may name only these |
| account | the one named like the repository owner (`gh auth token --user OWNER`), otherwise the active account |
| limits | 4 MiB of output, 60 s per command (coreutils `timeout` ends gh even if the broker dies) and 4 commands at a time; 16 connections, 10 s to send a request; token-like strings are cut from the output |
| log | `~/.pi/agent-local/gh-broker.log`, one line per request, allowed or refused; the sandbox cannot reach it |

`gh help` in the sandbox lists what is available. `jq` is there, so pipe the output through it instead of `--jq`. `SPI_GH=0 spi`
starts without the broker. `--jq` and `--template` are refused on purpose: the host `gh` would evaluate them with the token in
its environment.

What stays open: the model can read whatever the token can read in those repositories (code, pull request text, comments, CI
logs), and text from there is as untrusted as any file it reads; it cannot write to GitHub and has nowhere to send the data. The
list of repositories comes from the folder's own `.git/config`, so a cloned repository that names other repositories as remotes
makes those readable too.

Why bubblewrap and not a Docker image: Docker needs a root daemon (here it is switched off, and the `docker` group is root in
disguise). Rootless podman with an image mounting the folder gives the same kernel isolation (user, mount, pid and network
namespaces), at the price of an image to build and update, SELinux relabelling of the mounted folder and a way to reach the
server; it pays off only when the sandbox needs another distribution or other tool versions than the host. bubblewrap uses
the host's own `/usr` read-only, starts in milliseconds and has no image to keep current; extra toolchains come with `SPI_RO`.
Both share the host kernel; a micro-VM (the Gondolin example of pi's own guide) is the step up if a kernel exploit by the
model is part of what you fear.

Install and use:

```sh
install -m 755 tools/spi tools/spi_gh_broker.py tools/spi_gh_shim.py ~/.local/bin/    # the two .py files stay next to spi
mkdir -p ~/.pi/agent-local                  # put models.json (above) in it, and this settings.json:
```

```json
{
  "defaultProvider": "strata",
  "defaultModel": "qwen3.6-35b-a3b",
  "modelThinkingLevels": { "strata/qwen3.6-35b-a3b": "off" },
  "enableInstallTelemetry": false,
  "compaction": { "enabled": true, "reserveTokens": 8192, "keepRecentTokens": 12000 }
}
```

```sh
cd ~/some/project && spi               # interactive, the local model
spi -p "list the files here"           # one shot
spi --shell                            # a shell in the same sandbox: see what the model sees
SPI_RO=/home/you/.cargo:/home/you/go spi   # more directories, read-only (toolchains the project needs)
SPI_GH_REPOS=owner/other-repo spi      # more repositories for the read-only gh
```

What it does not stop: the model can still delete or rewrite everything in the folder you started it in, so commit or stash
first and look at the diff afterwards (files it leaves there, such as a `Makefile` or a script, run later on the host if you
run them); the names of the directories on the way down and the read-only system under `/usr` stay visible; inside the folder
git works from its root only, because the repository above a sub-folder is out of reach; it has no git identity and no credentials, so you commit and push yourself, outside.
Without a network it cannot install packages. If you start a plain `pi` while the local model is its default in
`~/.pi/agent/settings.json`, that run has no sandbox at all: use `spi` for the local model.

## Things that matter with a 64K window

- **Context files count.** pi reads `AGENTS.md` and `CLAUDE.md` from the working directory and every parent. The base
  prompt with the tool list is about 1.8K tokens; a large instruction file in a parent directory is added to every
  request and can exceed the window (the server then answers 400, "request exceeds the context window"). Start from a
  directory without such files, trim them, or run `pi --no-context-files`.
- **The start of a conversation is cached.** The recurrent layers cannot step back, so the engine keeps a checkpoint of its
  state at the start of the last turn (the system prompt and the tools, about 1.5K tokens) in host memory and on disk
  (`run/ckpt`, 84 MB each, six kept). A new pi session in a directory seen before reads only its own message (about 20
  tokens, 0.3 s instead of 18 s); the first session in a new directory reads everything once (9 s with the model in the
  page cache, 11-15 s from a cold disk), because pi puts the working directory at the end of its system prompt. A running conversation only reads its new tokens.
  Editing earlier messages, or compaction, invalidates the live state and the next turn reads from the nearest
  checkpoint or from the start, at the prompt speed (about 180-200 tokens/s with the model in the page cache, 150 from a
  cold disk; the CPU computes 96 of the 256 experts of each layer next to the card, `--cpu-experts`).
- **Scripts.** `pi -p` waits for its standard input to close: outside a terminal run it with `< /dev/null`, or it looks
  hung.
- **Greedy decoding.** The launcher sets greedy sampling for requests that give none, which lets the model's MTP block
  draft ahead (about 90% of the drafts are accepted). A request that sets its own temperature or penalties is decoded
  without drafts. `bin/strata-serve --sampling card` uses the model card's sampling for everything instead.

## The Strata extension for pi

`tools/pi-strata` is a pi extension for this server. A long recorded pi session (128 turns) showed where a 64K local model
loses its time, and none of it is the decoding speed:

- 67 calls in a row returned the same output (a `git diff | grep | head -N` with a different N each time): 51% of all
  tool-output tokens, and half of the session.
- Each compaction cost about nine minutes: the summary request reads its text from scratch (a different prompt, nothing
  in the cache), the summary is written token by token, and the next turn reads the kept tail again from scratch. pi
  keeps `keepRecentTokens` by its own estimate (about 4 characters per token); code is 2.8 characters per token, so
  12,000 kept tokens were about 22,000 to read.
- `read` returns the file without line numbers, so the model guesses `file:LINE` (1 of 27 references exact in the test below).

| Hook | What it does |
| --- | --- |
| `tool_result`, `tool_call` | Loop guard: an output identical to an earlier one of the same tool in the window is replaced by a two-line stub; after three identical results in a row from one command (numbers in its arguments ignored, `file1` and `file2` still differ) that command is refused, and after three refusals the run is stopped. `edit` and `write` are exempt, and an edit ends a run. |
| `tool_result` | Output cap: `bash` and other tools keep the head and the tail of a result over 6,000 characters, with the omitted line count and the command to get more. Escape codes and progress-bar rewrites are removed; whitespace and line endings (CRLF included) are never touched, so text copied into an edit still matches. |
| `tool_result` (`read`) | Lines are numbered from the real file line (`42<TAB>text`). A file over 24,000 characters keeps both ends and lists the definitions of the omitted part with their line numbers. |
| `tool_call` (`edit`) | Line-number prefixes are removed from `oldText` only when every line matches a line that a numbered read of that file showed (a TSV whose own lines start with a number and a tab is left alone), and then from `newText` too. |
| `session_before_compact` | Keeps about 2,500 estimated tokens instead of 12,000 (the cut moves only across plain messages; with a branch summary, a custom message or a context edit in between pi's own cut stays), builds a deterministic digest of the dropped part (the request, each action with the first line of its result, the model's notes), asks the model for a note of at most 300 words from that digest, and appends pi's `read-files` and `modified-files` lists. If the model gives nothing, the digest itself becomes the summary. |
| `before_agent_start` | Adds a short section to the system prompt: narrow commands, no repeated reads, numbered lines, batch independent lookups. |

Install (the sandbox of `tools/spi` mounts the folder read-only and `spi` warns on start when `extensions/strata` is missing; plain `pi` can load it with `pi -e tools/pi-strata/index.ts`):

```sh
mkdir -p ~/.pi/agent-local/extensions
cp -r tools/pi-strata ~/.pi/agent-local/extensions/strata
install -m 755 tools/spi tools/spi_gh_broker.py tools/spi_gh_shim.py ~/.local/bin/   # the copy that mounts extensions and gh
```

Settings go in an optional `config.json` next to `index.ts`: `guard` (`minChars`, `window`, `nudgeAt`, `blockAt`,
`abortAfterBlocks`, `exempt`), `output` (`bashChars`, `otherChars`), `read` (`lineNumbers`, `maxChars`), `compaction` (`enabled`,
`keepRecentEstimatedTokens`, `digestChars`, `summaryMaxTokens`), `promptGuidance`. Each part can be switched off alone, for
example `{"read": {"lineNumbers": false}}`.

Measured on this server (window 40,000, the task "review nine modules one at a time and append a paragraph to `notes.md`
for each", same model, greedy):

| Run | Model time | Prompt read | Generated | Exact `file:LINE` references |
| --- | --- | --- | --- | --- |
| pi alone | 776 s | 64.7K tokens | 3,850 | 1 of 27 |
| extension, compaction only | 775 s | 59.9K | 3,610 | 0 of 27 |
| extension, all parts | 651 s | 52.8K | 3,367 | 26 of 27 |

Compaction alone: the summary request read 992 tokens instead of 3,953 and the first turn after it 5,369 instead of 8,708
(6.4K against 12.6K tokens read in total). The loop guard, replayed over the recorded 128-turn session, refuses the first
repeat at turn 66 and stops the run at turn 68; the 60 turns after it would not have happened. What the numbers do not show:
how a different model behaves, and the loop guard has only been run on that one recording plus unit tests.

Live check of the loop guard: seven commands that return the same output. pi alone returned seven full copies; with the
extension the first was full, the next three became stubs, the next two were refused and the seventh stopped the run. Two
more tasks (six questions about the code, a constant rename with `edit`) gave the same answers and a correct edit with and
without the extension.

Trade-offs: line numbers add about 3 tokens per line, 19% more tokens on the six files measured (13% to 30%), to every
read; the cap and the refusal can hide something the model wanted (the marker says how to get it); a note of 300 words
keeps less than a long summary.

Checks: `cd tools/pi-strata && node --test test/*.test.ts` (unit tests), `node scripts/replay-guard.ts session.jsonl`
(what the guard would do to a recorded session, totals only).

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `ECONNREFUSED` or a connection error on `127.0.0.1:8080` | the server is not running: start `bin/strata-serve`; for another address change `baseUrl` |
| empty reply, 400 in the server log | the request is larger than the window, see context files above |
| first reply after a long pause is slow | the page cache lost the experts; speed returns after a few requests |
