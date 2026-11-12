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
| no network of its own, except a bridge to the local model server on `127.0.0.1:8080` | `--unshare-net` plus `socat` over a socket to the host ([below](#local-models-and-switching-between-them)) |
| the public web, through a filtering proxy: public addresses only, any port but the mail ports; nothing on the host, in the local network or behind the VPN | a proxy on the host behind another socket ([below](#web-access-in-the-sandbox)) |
| `gh`, read-only, for the repositories of the folder: the token stays outside | a broker on the host behind a second socket ([below](#gh-in-the-sandbox)) |
| its own profile, `models.json` and `settings.json` read-only, sessions kept; an empty `auth.json` that cannot be written | `~/.pi/agent-local` |

It refuses to start in `/`, in your home directory itself or in a hidden folder of it, and starts the local model server or checks that one answers.
Checked here: from the sandbox the home canary, a sibling folder, `~/.ssh` and `auth.json` do not exist, `touch /usr/bin/x`,
`rm /usr/bin/true` and `echo x > /etc/passwd` fail with "Read-only file system", `sudo` refuses, `curl --noproxy '*'
https://example.com` cannot resolve a name, and `rm -rf --no-preserve-root /` removes nothing outside the mounted folder (the host's `/usr/bin`,
canary files and secrets were intact afterwards).

### Local models, and switching between them

`spi` runs local models only; plain `pi` outside it keeps every provider. In `spi`:

- `--provider`, `--api-key`, `--extension` (`-e`), `--no-extensions` (`-ne`) and `--approve` (`-a`) are refused, and so are the commands
  `pi install`, `remove`, `uninstall`, `update`, `auth`, `mcp` and `config`;
- `auth.json` is an empty file that cannot be written, so `/login` has nowhere to keep a key, and the environment is empty, so no key
  comes from a variable;
- `models.json` lists the local models only, and it is read-only;
- the Strata extension, which `spi` switches on with `SPI_LOCAL_ONLY=1`, empties the catalog of every provider whose address is not
  `127.0.0.1`, `localhost` or `[::1]` (the built-in ones and those of other extensions, `/model`, `/scoped-models` and `Ctrl+P` included),
  refuses `/login` and `/logout`, switches back from a non-local model and stops a request that would go to one, and does not trust
  project-local extensions, skills, prompts and settings of the folder (`AGENTS.md` is still read);
- the web proxy refuses the API and sign-in hosts of the known model providers (`PROVIDER_HOST_PATTERNS` in `tools/spi_web_proxy.py`).
  The list cannot be complete: it narrows the way to a hosted model and does not close it. A second `pi` that the model starts from
  its shell without the extension (`PI_CODING_AGENT_DIR=$(mktemp -d) pi -ne`), or a `curl` with a key it was given, is outside what the
  extension can stop; only the proxy list and the empty environment stand in the way.

When no server answers on port 8080, `spi` starts the one of a local model through the model manager (`tools/spi_models.py`) and stops
it when `spi` ends. The models it may start are listed in `~/.pi/agent-local/local-models.json` (the path can be changed with
`SPI_MODELS`):

```json
{
  "root": "~/path/to/strata",
  "default": "qwen3.6-35b-a3b-q5k",
  "models": {
    "qwen3.6-35b-a3b-q5k": {"file": "models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS-mtp-d5.gguf"},
    "qwen3.6-35b-a3b-iq4xs": {"file": "models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS-mtp.gguf", "args": ["--experts-used", "8", "--expert-prefetch", "0"]},
    "qwen3.6-35b-a3b-q4kxl": {"file": "models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "args": ["--experts-used", "8", "--expert-prefetch", "0"]}
  }
}
```

A model id is the `id` of an entry in `models.json`, so `/model` in pi lists the same models. The first one (`default`) starts with `spi`.
The dense Q5_K file is the default because it runs the measured fast setting of the launcher (7 experts per token, a prefetch of 3; see
"The default stack" in the README); the two entries with `--experts-used 8 --expert-prefetch 0` run their files with the model's own
routing and no prefetch, which is what to pick when the output must not differ from the original model's. When another model is chosen with `/model` or `Ctrl+P`, the next request makes the manager finish the request that is running,
stop the server, start the other one and pass the request on: the first answer after a switch waits for the load, about ten seconds
with the files in the page cache and longer from the SSD. Only one model is in memory at a time. The manager answers `GET /v1/models`
itself (the local models and which one is loaded), replaces the `model` of a request with the name the server knows itself by, and
refuses a model that is not in the list with a 404.

The server of the manager listens on `127.0.0.1:8081` (`SPI_SERVER_PORT`), so it does not meet a server that you started yourself;
its output is in `server.log` and the manager's own in `models.log`, both in the profile folder. A server that already answers on
port 8080 is used as it is: it keeps its own model, and `/model` cannot switch it. `STRATA_ROOT` names the Strata folder when the config
has no `"root"`; `spi` run from `tools/` of the checkout finds it by itself. When `spi` ends, it waits for the server to stop, which
puts the CPU frequency cap back.

### gh in the sandbox

`gh` inside the sandbox has no token, so by itself it only says "not logged in". `spi` starts a small broker on
the host (`tools/spi_gh_broker.py`) and puts a `gh` in front of the real one that sends each command to it over a unix socket.
The broker checks the command against a fixed list, runs the real `gh` with your token and returns the output. The token never
enters the sandbox.

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
logs), and text from there is as untrusted as any file it reads; it cannot write to GitHub, but it can send what it read to the public web through the proxy below. The
list of repositories comes from the folder's own `.git/config`, so a cloned repository that names other repositories as remotes
makes those readable too.

### Web access in the sandbox

The sandbox has no network interface of its own, so `curl https://example.com` needs the proxy. `spi` starts a small web proxy on
the host (`tools/spi_web_proxy.py`) and bridges it into the sandbox on `127.0.0.1:3128`; `HTTP_PROXY`, `HTTPS_PROXY` and
`NODE_USE_ENV_PROXY` point at it, and `NO_PROXY` keeps `127.0.0.1` direct, so pi still reaches the model without it. `curl`, `git`
over https, `npm`, `pip` and node's `fetch` work through it.

| | |
| --- | --- |
| accepted | `CONNECT host:port` and plain `http://` requests with a full address; nothing else |
| addresses | the proxy resolves the name on the host and refuses the request unless every answer is a public address: loopback, private, link-local, carrier-grade NAT, multicast and reserved ranges are refused, and so are IPv6 forms that carry such an address (IPv4-mapped, 6to4, Teredo, NAT64). It then asks the kernel (`ip route get` for a TCP connection to that port, which honours policy rules and every routing table) how it would reach each address and keeps only those it reaches the way it reaches a documentation address, along the default route; an address of this host, an unreachable one (IPv6 on a host without it) and one behind a route other than the default are left out, and a failed lookup leaves the address out too. A route can change between the lookup and the connection, so the source address of the connected socket, which the kernel takes from the route it really used, is compared with the default route's as well. A split-tunnel VPN sends some public ranges through the company network: whole provider and CDN blocks (Cloudflare, a cloud's `/8`) and, next to them, single company endpoints. The two cannot be told apart by address alone, so the width of the route decides (`SPI_WEB_VPN`): by default (`hosts`) a route of `/22` or wider counts as the internet and a route of `/23` or narrower (down to a single address) as a company host, which is refused; `block` refuses every address that the VPN routes, `allow` none. With the default the agent can reach a company service that sits in a wide provider block and is open to the company's addresses, and with `block` it cannot reach a Cloudflare-hosted site while the VPN is up (see below for what is done about it). It connects to the address it checked, so a changed DNS answer cannot redirect it, and it follows no redirects. This keeps the local services of the host (a database on `127.0.0.1`, ssh) and the networks behind the VPN out of reach. A name that does not resolve and a name that resolves inside get the same answer, so the proxy cannot be used to probe internal names |
| ports | any, except the mail ports 25, 465 and 587; `SPI_WEB_PORTS=80,443` limits the list |
| needs | python3 and `ip` (iproute2) on the host; the proxy refuses to start without them |
| limits | 64 connections (another 64 are refused with a 503, the rest dropped), 10 s for the request head, 16 name lookups and 8 route lookups at a time, the first 16 addresses of a name, 120 s idle, one hour per connection |
| log | `~/.pi/agent-local/web-proxy.log`, one line per connection with the target, the bytes and the verdict; the sandbox cannot reach it |

Not covered: ssh, raw TCP and UDP (git over ssh, database clients) and tools that ignore the proxy variables. `SPI_WEB=0 spi`
starts without any web access.

A site may still refuse the sandbox: a shop behind Cloudflare answers a program with a browser check ("Just a moment...", HTTP
403) and gives its page only to a real browser. Nothing here works around that, and the `web_fetch` tool of the extension says so
instead of guessing.

What this gives away: the model, and anything it runs, can send data to any public address on those ports, so a text from a web
page that steers it can also move out what it can read, which is the content of the folder, `.env` files included. The sandbox
still has no token, no keys and no write access outside the folder, but keep secrets out of the folder or run with `SPI_WEB=0`
when the folder holds some.

Why bubblewrap and not a Docker image: Docker needs a root daemon (here it is switched off, and the `docker` group is root in
disguise). Rootless podman with an image mounting the folder gives the same kernel isolation (user, mount, pid and network
namespaces), at the price of an image to build and update, SELinux relabelling of the mounted folder and a way to reach the
server; it pays off only when the sandbox needs another distribution or other tool versions than the host. bubblewrap uses
the host's own `/usr` read-only, starts in milliseconds and has no image to keep current; extra toolchains come with `SPI_RO`.
Both share the host kernel; a micro-VM (the Gondolin example of pi's own guide) is the step up if a kernel exploit by the
model is part of what you fear.

Install and use:

```sh
install -m 755 tools/spi tools/spi_gh_broker.py tools/spi_gh_shim.py tools/spi_web_proxy.py tools/spi_models.py ~/.local/bin/    # the .py files stay next to spi
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
SPI_WEB=0 spi                          # no web access at all
SPI_PRIORITY=normal spi                # pi and its commands at the normal priority (the default: a control group with weight 20 against 100)
SPI_WEB_VPN=block spi                  # also refuse the wide public ranges that a VPN routes (see the proxy section)
```

What it does not stop: the model can still delete or rewrite everything in the folder you started it in, so commit or stash
first and look at the diff afterwards (files it leaves there, such as a `Makefile` or a script, run later on the host if you
run them); the names of the directories on the way down and the read-only system under `/usr` stay visible; inside the folder
git works from its root only, because the repository above a sub-folder is out of reach; it has no git identity and no credentials, so you commit and push yourself, outside; it has no CPU, memory or process limit of its own, so a runaway command can slow the machine until it is stopped or `spi` ends.
If you start a plain `pi` while the local model is its default in
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
| `before_agent_start` | Adds a short section to the system prompt: narrow commands, no repeated reads, numbered lines, batch independent lookups, that the web is reachable, and not to retry a site that refuses. |
| `tool_result` (closed doors) | Tells a model that keeps knocking on a site that turns it away that this is pointless and where else to look. The sites of a call are the URL of `web_fetch` or the addresses in a `curl` or `wget` command (`www.`, the port, case, a trailing dot and an IDN spelling do not matter; a result is credited to the final address after redirects, and a command with several addresses is not credited to any). A result counts as a refusal when its first significant status is 403 (the proxy's "Connection established" line and redirects are skipped, and a page that shows `HTTP/2 403` in its text after a 200 does not count), or, when it has no status, the title is "Just a moment..." or "Access Denied", or the web proxy refused the address, or the name does not resolve; 401 (a login) and 429 (a pause) do not count. The refused result gets a note at its end: the site turns programs away, every page of it will do the same, and the ways round it that work from the sandbox (a Bing search through `web_fetch`, the maker's or another shop's site, a price-comparison site, an API or a feed, `web.archive.org`, or asking the user for the text); from the second refusal the note says how many times it was turned away. Nothing is blocked by default. `"doors": {"block": true}` in `config.json` also stops the calls that name a site after `refusalsBeforeBlock` (2) refusals for `blockMinutes` (30), a name inside the command text included (`host=shop.test; curl "https://$host/"`); `"enabled": false` turns the notes off. The count is dropped when the site answers normally with some output and at a new session. |
| tool `web_fetch` | Opens an http or https URL and returns it as text: a page is reduced to its readable text (the `<main>` or `<article>` part when it has real content, without scripts, styles, navigation and blank lines, and with a run of eight or more short list items joined into one line), JSON and plain text come as they are, a charset is honoured, redirects are followed, at most 2 MB are read. A result is about 5,000 characters and names the `offset` for the next part, so a long page is read in pieces. A Cloudflare browser check, a refused address, a missing host and a timeout are reported in words. Turn it off with `"web": {"enabled": false}` in `config.json`. Inside `spi` it goes through the web proxy; plain `pi` fetches directly. |

Install (the sandbox of `tools/spi` mounts the folder read-only and `spi` does not start pi when `extensions/strata` is missing, because that extension keeps it to local models; plain `pi` can load it with `pi -e tools/pi-strata/index.ts`, where it leaves every provider alone unless `SPI_LOCAL_ONLY=1` is set):

```sh
mkdir -p ~/.pi/agent-local/extensions
cp -r tools/pi-strata ~/.pi/agent-local/extensions/strata
install -m 755 tools/spi tools/spi_gh_broker.py tools/spi_gh_shim.py tools/spi_web_proxy.py tools/spi_models.py ~/.local/bin/   # the copy that mounts extensions, gh and the proxy
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
| `ECONNREFUSED` or a connection error on `127.0.0.1:8080` | the server is not running: start `bin/strata-serve`, or run `spi` with a `local-models.json`; for another address change `baseUrl` |
| `spi: the model manager did not start` | the config is missing or wrong, or the Strata folder is unknown: read the message above it and `models.log` |
| a 503 from the manager, "already taken" or "ended with code" | port 8081 is used by another program, or the server of the model failed to start: see `server.log` |
| empty reply, 400 in the server log | the request is larger than the window, see context files above |
| first reply after a long pause is slow | the page cache lost the experts; speed returns after a few requests |
