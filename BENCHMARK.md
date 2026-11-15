# Benchmark: maskerade vs. other masking tools on pentest traffic

Measured 8 November 2026 with maskerade 0.1.6.

Every tool ran with its strongest documented configuration, on the same laptop CPU, on the same inputs. All of them were scored by the same harness.

**The short version:**

- On HTTP traffic from web apps, maskerade masked **95% of sensitive values**. The next-best tool masked 72%.
- maskerade also over-masked the least and broke **none of the 348 JSON bodies**. The other tools broke between 9 and 56.
- maskerade is built for pentest data, not for general prose PII, and these results only cover pentest data.

## What was measured

**HTTP corpus.**

- 1,060 requests and responses captured from 36 open-source web apps running in a test lab.
- The apps include Juice Shop, crAPI, DVWA, WebGoat, Keycloak, Grafana, Portainer, Directus, NocoDB and Prometheus.
- 1,909 annotated values: target hosts, file paths, session IDs, CSRF tokens, JWTs, emails, passwords, usernames, person names, API keys and other identifiers.
- 348 of the messages have a JSON body.
- 364 messages contain nothing sensitive.

**Pentest prompts.**

- 30 prompts written the way a pentester asks Claude for help.
- 435 annotated values: names, companies, internal hostnames, credentials, IDs.

**The HTTP corpus is public.** Download `http-corpus-v1.2.tar.gz` from the [corpus release](https://github.com/adamectimotei/maskerade-proxy/releases/tag/corpus-v1.2) and check its SHA-256 against the release notes.

- It was captured from the 36 open-source apps running in a local lab. All accounts and personal data in it are synthetic: names, emails on `example.test`, passwords, IBANs and card numbers are generated.
- Each pair's `metadata.json` names the app's repository and commit.
- `GOLD_README.md` describes the annotation format, and `PATCH_CHANGELOG.md` lists every revision to the annotations.

**Licence:** the annotations are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You can reuse them, including commercially, if you credit "Timotei Adamec, maskerade HTTP corpus v1.2" and link the release. The captured traffic itself belongs to the apps it came from, under their own licences.

**The prompt set stays private,** because it is derived from real engagements.

## Tools and settings

| tool | version | configuration |
|---|---|---|
| **maskerade**, fast | 0.1.6 | default; no rules file |
| **maskerade**, full | 0.1.6 | `--tier full` (adds the GLiNER2 PII model); no rules file |
| PrivAiTe, `max` | 0.8.0 | Presidio + OpenAI privacy-filter (ONNX q4) + GLiNER |
| PrivAiTe, `onnx` (default) | 0.8.0 | Presidio + OpenAI privacy-filter (ONNX q4) |
| PrivAiTe, `minilm` | 0.8.0 | Presidio + PrivAiTe's own MiniLM PII model |
| LiteLLM + Presidio | litellm 1.104.2 | Presidio guardrail, `pre_call`, all recognizers, spaCy `en_core_web_lg` |
| LiteLLM + Presidio, reversible | litellm 1.104.2 | as above, plus `output_parse_pii: true` |
| maskflow | maskflow-sdk 0.9.2 | all packs, spaCy `en_core_web_sm`, empty config |

**How each tool was run:**

- Each tool was used through its real integration path: the LiteLLM proxy, PrivAiTe's request pipeline, and the maskflow and maskerade SDKs.
- The output that reached the model was captured, and the masked values were recovered by aligning that output with the input. Every tool was scored the same way.
- All tools ran on CPU only, with 8 threads, on an Intel i7-1365U laptop under WSL2. No tool got a GPU.
- Every tool ran in English. maskerade had no rules file, so it ran without the client names and domains it is meant to use.
- The runner reuses the worker protocol from [privaite-bench](https://github.com/crp4222/privaite-bench). The scoring harness is not published.

## Results: HTTP corpus

1,909 annotated values in 1,060 messages.

| | recall | masked completely | over-masks | harmless messages masked | broken JSON | fused placeholders | collisions | p50 / p95 per message |
|---|---|---|---|---|---|---|---|---|
| **maskerade fast** | **94.8%** | **94.5%** | **1,587** | **142 / 364** | **0 / 348** | **0** | **0** | **2 ms / 32 ms** |
| **maskerade full** | **95.7%** | **95.1%** | 2,068 | 158 / 364 | **0 / 348** | **0** | **0** | 0.8 s / 10.8 s |
| PrivAiTe max | 72.1% | 55.7% | 4,217 | 250 / 364 | 38 / 348 | 76 | 2 | 1.4 s / 24.7 s |
| PrivAiTe minilm | 70.4% | 40.0% | 4,764 | 353 / 363 | 21 / 348 | 115 | 9 | 0.1 s / 2.3 s |
| PrivAiTe onnx | 69.5% | 41.1% | 3,294 | 242 / 364 | 28 / 348 | 77 | 2 | 0.3 s / 5.4 s |
| maskflow | 54.4% | 19.0% | 5,837 | 243 / 364 | 9 / 348 | 59 | 7 | 17 ms / 186 ms |
| LiteLLM + Presidio | 45.5% | 6.4% | 11,924 | 275 / 364 | 41 / 348 | 55 | 1,538 | 41 ms / 236 ms |
| LiteLLM + Presidio, reversible | 45.5% | 12.9% | 5,050 | 209 / 364 | 56 / 348 | 0 | 0 | 40 ms / 267 ms |

PrivAiTe minilm refused 2 of the 1,060 messages with a processing error.

**Two caveats on the recall figures:**

- **Target hosts (530 values) were masked 100% by every tool**, which compresses the gaps between them.
- **File paths (384 values) have no matching type in the other tools**, which widens maskerade's lead.

With both excluded, recall is maskerade fast **90.2%** and full **91.8%**, against PrivAiTe max and minilm 81.7%, PrivAiTe onnx 80.0%, maskflow 48.4% and LiteLLM 25.1%.

### Where maskerade loses

| type | maskerade fast / full | best other tool |
|---|---|---|
| Customer IDs (10) | 0 / 0 | 10 (all PrivAiTe presets) |
| App-specific numeric IDs (87) | 23 / 25 | 39 (PrivAiTe onnx, max) |
| Person names (38) | 20 / 34 | 38 (PrivAiTe max) |

PrivAiTe's models learn from context that the number next to "customer" or "account" is an identifier. maskerade looks at patterns, randomness and key names, so it masks a bare `48213` only when the key says it is an ID. Without its model, the fast tier also misses about half of the free-text person names.

## Results: pentest prompts

435 annotated values in 30 prompts.

| | recall | masked completely | placeholders that hit a real value | over-masks |
|---|---|---|---|---|
| **maskerade full** | **92.0%** | **88.7%** | 78.9% | 107 |
| **maskerade fast** | 81.6% | 79.3% | 81.4% | 81 |
| PrivAiTe minilm | 73.6% | 67.4% | 57.7% | 235 |
| PrivAiTe max | 71.3% | 68.3% | 53.0% | 275 |
| PrivAiTe onnx | 69.4% | 65.8% | 55.3% | 244 |
| LiteLLM + Presidio, reversible | 62.1% | 60.9% | 84.1% | 51 |
| LiteLLM + Presidio | 61.8% | 46.7% | 46.3% | 312 |
| maskflow | 34.0% | 26.0% | 50.9% | 143 |

## Definitions

- **Recall:** a value counts as masked when its text no longer appears in what was sent.
- **Masked completely:** every word of a multi-word value was masked. `Jan [PERSON_1]` counts as masked, but not completely.
- **Over-mask:** a masked value that matches no annotated value exactly.
- **Harmless messages masked:** messages with nothing sensitive in them that still got at least one placeholder.
- **Broken JSON:** a body that parsed before masking and does not parse after it.
- **Fused placeholder:** several values merged into one placeholder, so the model can't tell them apart.
- **Collision:** one placeholder standing for several different values. This happens when a tool uses unnumbered placeholders such as `<PERSON>`.
- **Why this differs slightly from the README:** the README's numbers come from maskerade's own scorer, which reads maskerade's internal mask map. This benchmark recovers what each tool masked from the text that was actually sent, because that's the only way to treat every tool the same. The two methods give identical recall. maskerade's precision and over-mask counts differ by less than 1 point (HTTP: 53.1% vs 53.3% fast, 46.7% vs 46.9% full; prompts, full tier: 79.8% vs 78.9%).
- **Fused to a letter:** a placeholder directly touching a letter, for example a value masked in the middle of a word.

  maskerade has 4 such messages:
  - 2 Trilium responses, where an IPv6-shaped match ended inside a hex token;
  - 1 Juice Shop response, with a phone number directly after a letter;
  - 1 Keycloak response, with a credential directly before a letter.

  The other tools have 77 to 429 such messages.

## Limits of this benchmark

- **I built both test sets, using maskerade's own type names.** The prompts were also used while developing maskerade. The competitors' types were mapped onto them, so treat these numbers as indicative, not independent. The HTTP corpus is public so you can check the annotations yourself.
- **The data is web-app traffic and pentest prompts only.** On general prose PII in seven languages, measured with someone else's data and labels on privaite-bench, maskerade full is mid-table and the fast tier is near the bottom. That isn't what maskerade is for.
- **No tool got custom rules.** In real use, maskerade's rules file holds the client's names and domains, which only adds to these numbers.
- **One run, one machine.** Latency depends on hardware. The full tier and the model-based tools are much faster on a GPU.
