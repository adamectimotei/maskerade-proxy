# maskerade

**A local masking proxy for the Anthropic API.** It runs on your own machine, replaces client data in your prompts and traffic with typed placeholders before anything is sent, and puts the real values back into the reply.

Built for penetration testers who use Claude (Claude Code, the SDKs, anything that speaks the Messages API) on client work and must not send client data to a third party.

```
your client (Claude Code, SDK)        maskerade, on 127.0.0.1         api.anthropic.com
──────────────────────────────        ──────────────────────────      ─────────────────
"admin is jan.novak@acmebank.cz"  ──▶  detect + mask            ──▶  "admin is [EMAIL_ADDRESS_1]"
"reset it for jan.novak@acmebank.cz" ◀── restore real values    ◀──  "reset it for [EMAIL_ADDRESS_1]"
```

> **Status: alpha (0.1.5).** It reduces what leaves your machine; it does not guarantee that nothing sensitive does. Read [What it does not do](#what-it-does-not-do) before relying on it, and keep reviewing what you send.

---

## How it works

- **Detects** client data with deterministic layers: your own rules (the client's names, domains and tenant slugs), around 100 regex patterns for credentials, tokens, keys, hashes, IPs, hostnames, emails and national IDs, checksum validators (IBAN, card numbers, national ID numbers), key/value structure (`"password": ...`), identifier shapes, and a randomness test for secrets nobody has a pattern for. The optional NER tier adds a local GLiNER model for names and organisations.
- **Masks** each value with a typed placeholder such as `[INTERNAL_HOSTNAME_1]`. The same value gets the same placeholder for the whole conversation, so the model can still reason about it ("`[USER_ID_1]` can read `[USER_ID_2]`'s invoices").
- **Forwards only the masked request** to `api.anthropic.com`, and appends a short note telling the model what the placeholders mean.
- **Restores** the real values in the reply, including streamed replies and tool-call arguments, so your client sees the real text.
- **Fails closed.** If a detector cannot run, or a detected value cannot be masked, the request is refused, never forwarded unmasked.

Everything runs locally on the CPU. No GPU, no model server, no account.

## Supported clients

| Client | Through maskerade? |
|---|---|
| **`claude` command-line tool (Claude Code)** | **Yes**, with a Claude subscription login or an API key. This is the main use; see [Quick start](#quick-start-with-claude-code). |
| **Claude desktop app in third-party mode** | **Yes**, with an Anthropic API key only (subscriptions don't work in this mode). See [Claude desktop app](#claude-desktop-app). |
| Claude desktop app signed in with a Claude account | **No.** Chat, Cowork and the Code tab all talk to Anthropic directly, even in a folder `maskerade init` prepared. |
| claude.ai in a browser | **No.** |
| Anything else that speaks the Messages API (SDKs, scripts) | **Yes**, by setting `ANTHROPIC_BASE_URL`. |

## Install

Python 3.11 or newer.

```bash
pip install maskerade
python -m maskerade.warm          # one-time: fetches the 1.6 MB BPE encoder, checksum-verified
maskerade --version
```

`warm` is required: the entropy detector needs that encoder, and without it the proxy refuses to start rather than run with a weaker detector. Behind a proxy, set `HTTPS_PROXY` for this command; for a machine with no internet access at all, see the end of [Command reference](#command-reference).

### Full tier (adds the NER model)

The `ner` extra pulls in PyTorch and the GLiNER model, a few GB in total. Install the **CPU** build of PyTorch first; otherwise pip fetches the much larger CUDA build, which this tool cannot use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "maskerade[ner]"
python -m maskerade.warm --full   # one-time: encoder + model (~1.2 GB)
```

Run it with `--tier full`. Startup takes noticeably longer while the model loads.

### Docker

Images are published to `ghcr.io/adamectimotei/maskerade`: `0.1.5` (fast tier, ~190 MB) and `0.1.5-ner` (full tier, model included, ~2.6 GB). See [Docker](#docker) below before running one: **how you publish the port matters.**

## Quick start with Claude Code

From the directory you'll work in for the engagement:

```bash
maskerade init --engagement eng-2026-014
```

`init` asks for the client's names and domains and writes them to `.maskerade/rules.txt`. It points the `claude` command-line tool at the proxy through `.claude/settings.local.json` (the desktop app ignores that file; see [Supported clients](#supported-clients)), and warns you about places the client's name already appears in clear text (the folder path, git remote, branch). It writes no credentials anywhere.

Then start the proxy **from that same directory** and use Claude Code as usual:

```bash
maskerade proxy --engagement eng-2026-014            # fast tier (default)
maskerade proxy --engagement eng-2026-014 --tier full
```

For any other client, point it at the proxy yourself:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787            # bash
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"          # PowerShell
```

Your API key, or your Claude subscription login, works unchanged; the proxy passes your credentials through and never stores or logs them.

### Claude desktop app

Only in the app's **third-party mode**, which sends Chat, Cowork and Code to a gateway you choose, and only with an **Anthropic API key**: usage is billed to the API, not to a Claude subscription. Tested with the Code tab.

1. Start the proxy from the engagement directory, as above.
2. In the app: **Help → Troubleshooting → Enable Developer Mode**, then **Developer → Configure Third-Party Inference…**
3. Under **Connection**, set:
   - **Inference provider:** Gateway
   - **Gateway base URL:** `http://127.0.0.1:8787`
   - **Gateway API key:** your Anthropic API key
   - **Gateway auth scheme:** x-api-key
4. **Apply Changes**, then **Save & Restart**, and on the sign-in screen start with the third-party configuration.

To go back to your normal setup, choose the Anthropic sign-in on the sign-in screen. Signed in that way, nothing in the app goes through maskerade.

## Custom rules: the client's own identity

No detector can know your client's name in advance, so tell it. `.maskerade/rules.txt` is one value per line:

```text
# acme bank, engagement 2026-014
acmebank.cz
Acme Bank a.s.   = CLIENT_NAME
ACME-PAYROLL     = CLIENT_SYSTEM
```

- A bare line becomes `[CLIENT_1]`, `[CLIENT_2]`, ...; `value = TYPE` picks the type.
- Matching is case-insensitive and on word boundaries. A domain covers all its subdomains, so `acmebank.cz` also masks `portal.acmebank.cz`.
- An email address on a client domain is masked whole, as `[EMAIL_ADDRESS_n]`, never cut in half by a rule.
- The proxy re-reads the file when it changes. Check a file before trusting it:

```bash
maskerade rules check .maskerade/rules.txt --against sample.txt
```

## Tiers

| | fast (default) | full (`--tier full`) |
|---|---|---|
| Install | `pip install maskerade` | `maskerade[ner]` + CPU PyTorch |
| Detectors | custom rules, ~100 regex patterns, entropy/format, key/value, checksums, identifier shapes | all of fast, plus GLiNER (names, organisations, locations) with an evidence gate for organisation names |
| Speed | milliseconds per message | slower; the model runs on CPU |
| Catches less of | free-text person and company names your rules don't list | very large bodies: the model skips inputs over 16 384 characters, the other detectors still run |

### Measured detection (0.1.0, 25 Oct 2026)

Measured with **no rules file**, which is the worst case; your rules only add to this. Two test sets, both private:

- 30 annotated pentest prompts, 435 values: names, companies, hostnames, credentials, IDs
- 1,060 HTTP requests and responses from open-source web apps, 1,909 values. 364 of these messages contain nothing sensitive.

| | fast | full |
|---|---|---|
| Prompts: values masked | 81.6% | 92.0% |
| Prompts: masked completely | 79.3% | 88.7% |
| HTTP traffic: values masked | 94.8% | 95.7% |
| HTTP traffic: masked completely | 94.5% | 95.1% |
| Prompts: placeholders that hit a real value | 81.4% | 79.7% |
| HTTP traffic: placeholders that hit a real value | 52.8% | 46.3% |
| Time per message, laptop CPU (i7-1365U) | milliseconds | ~1.5 s for traffic, ~3-5 s for a prompt; up to ~20 s for large bodies |

*Masked completely* also requires every word of a multi-word value to be masked: `Jan [PERSON_1]` counts as masked but not as completely masked.

**What it misses most:**

- In HTTP traffic, application-specific identifiers with no recognisable shape. These are 72 of the 83 misses.
- In prompts, internal hostnames without a `.corp`/`.local`-style suffix.
- On the fast tier, free-text person and company names: only 38% are caught without the model.

**Over-masking is deliberate:** 160 of the 364 harmless HTTP messages still got at least one placeholder.

**Caveat:** both test sets were built by the author, and the prompt set was also used during development. Treat these numbers as indicative, not independent. Put everything you know about the client in your rules file.

## Is it working?

- **Startup banner.** The proxy prints the bind address, the detector set (`mode`), where mask maps live, the engagement and which rules file it loaded. `custom rule none` means no rules file was found.
- **Every response carries headers:**
  - `X-Maskerade-Mode`: the detectors that ran
  - `X-Maskerade-Session`: the session, and `new` or `continued`
  - `X-Maskerade-Rules`: how many of your rules are loaded
  - `X-Maskerade-Unscanned`: request fields forwarded **without** inspection (see below)
  - `X-Maskerade-Refused`: set when a request was refused instead of sent
- **Health check:** `curl http://127.0.0.1:8787/healthz`
- **Loopback only:** `netstat -ano | findstr 8787` (Windows) or `ss -ltnp | grep 8787` should show `127.0.0.1:8787` and nothing else.

## Mask maps, sessions and retention

The mapping from placeholders to real values (the *mask map*) is stored per conversation, so the same value keeps the same placeholder across turns and the reply can be restored. **It is client data.** It lives in:

| OS | default location |
|---|---|
| Windows | `%LOCALAPPDATA%\maskerade` (deliberately not `Documents`, which OneDrive syncs) |
| Linux | `$XDG_STATE_HOME/maskerade`, or `~/.local/state/maskerade` |
| macOS | `~/Library/Application Support/maskerade` |

Set `MASKERADE_STATE_DIR` to put it elsewhere. The files are restricted to your user. On Linux and macOS the proxy refuses to start if it cannot do that; on Windows it prints a warning. It also warns if the location looks like a cloud-synced folder.

```bash
maskerade sessions list                          # ids, labels, placeholder COUNTS; never values
maskerade sessions purge --older-than 30d
maskerade sessions purge --session eng-2026-014  # one engagement
maskerade sessions purge --all
```

Sessions unused for 30 days are deleted when the proxy starts (`--retention-days` to change it).

## Docker

From the engagement directory, with [docker-compose.yml](https://github.com/adamectimotei/maskerade-proxy/blob/main/docker-compose.yml) from this repository:

```bash
export MASKERADE_ENGAGEMENT=eng-2026-014                  # bash
$env:MASKERADE_ENGAGEMENT = "eng-2026-014"                # PowerShell

docker compose -f path/to/docker-compose.yml --project-directory . up -d
```

The compose file refuses to start without `MASKERADE_ENGAGEMENT`. It mounts `./.maskerade` from the engagement directory read-only, so the container picks up the rules file that `maskerade init` writes there; without pip on the host, create `.maskerade/rules.txt` by hand (see [Custom rules](#custom-rules-the-clients-own-identity)). For the full tier, switch to the `-ner` image and add `--tier full`, as the comments in the file show.

The proxy refuses to listen anywhere but loopback unless given `--allow-remote`. Inside a container a loopback-only proxy is unreachable from the host, so **the image listens on all interfaces inside the container, and the port publish is what keeps it on your machine**:

```bash
docker run -p 127.0.0.1:8787:8787 ...   # correct: reachable from this machine only
docker run -p 8787:8787 ...             # WRONG: anyone on your network can use it
```

The proxy has no authentication, holds your mask maps and restores real values, so a wrong publish exposes client data to your whole network. The [compose file](https://github.com/adamectimotei/maskerade-proxy/blob/main/docker-compose.yml) gets this right and explains it. Also:

- Other containers on the **same Docker network** can reach the proxy. Don't attach it to networks shared with things you don't trust.
- Keep mask maps in a **named volume** (the compose file does). The proxy refuses a bind-mounted directory it cannot restrict to its own user.
- The first start downloads the 1.6 MB encoder into the volume. Later starts, including the NER image's, need no network beyond `api.anthropic.com`.
- The images are tested on Linux with Docker Engine. Docker Desktop on Windows and macOS is not tested yet.

## What it does not do

Read this before you depend on it.

- **Only message content is inspected.** The system prompt, tool definitions and several other request fields are forwarded **as-is**, because masking them breaks the client. Every response names them in `X-Maskerade-Unscanned`. For Claude Code these fields include your **working directory, home directory, git remote and branch, and your account name**, so work from a neutrally named directory (`maskerade init` checks for this).
- **Images and PDFs are forwarded unmasked.** Their bytes cannot be masked, so a pasted screenshot, or an image file the agent reads, reaches Anthropic as it is. They are named in `X-Maskerade-Unscanned` (`image`, `document`). The text of a plain-text document, and a document's title and context, are masked.
- **Detection is not complete.** A value no rule, pattern or model recognises goes through. The fast tier in particular misses free-text names that aren't in your rules.
- **It only proxies the Anthropic Messages API** (`/v1/messages`, `count_tokens`, `/v1/models`, and the OAuth token refresh). Any other path is refused.
- **It only covers traffic sent through it.** Your client's telemetry, update checks and other requests do not go through `ANTHROPIC_BASE_URL`. For Claude Code, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` (which `init` sets) turns most of them off. With a non-Anthropic base URL, Claude Code also disables some features, such as MCP tool search and Remote Control.
- **It over-masks on purpose.** Harmless values are sometimes masked too, because missing a real one costs far more. For example, `?email%3Dname%40example.com` becomes `?[EMAIL_ADDRESS_1]`, losing the parameter name along with the address.
- **No authentication.** Anyone who can reach the port can use it. Keep it on loopback.

## Command reference

```
maskerade init --engagement NAME [--client NAME ...] [--domain DOMAIN ...]
maskerade proxy [--engagement NAME] [--tier fast|full] [--port 8787]
                [--custom-rules FILE] [--config FILE] [--state-dir DIR]
                [--retention-days 30] [--no-system-note | --system-note FILE]
maskerade rules check FILE [--against SAMPLE]
maskerade sessions list | purge (--all | --older-than Nd | --session ID) [--yes]
maskerade --version
python -m maskerade.warm [--full] [--to DIR] [--check] [--traceback]
```

`--help` on any subcommand has the details. `--config FILE` replaces the packaged detection config rather than merging with it, so start from a copy of `maskerade/default_config.yaml`. For an offline machine, run `python -m maskerade.warm --full --to DIR` on a connected one, copy the folder over, and set the variables it prints.

## Security

Report vulnerabilities privately, as described in [SECURITY.md](https://github.com/adamectimotei/maskerade-proxy/blob/main/SECURITY.md). Never include real client data in a report.

## Licence

Apache-2.0. See [LICENSE](https://github.com/adamectimotei/maskerade-proxy/blob/main/LICENSE).

The BPE encoder (`cl100k_base`) is not redistributed here; `maskerade.warm` downloads it from its publisher. The full tier's model, `fastino/gliner2-privacy-filter-PII-multi`, is Apache-2.0 and is included in the `-ner` Docker image.
