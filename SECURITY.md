# Security policy

maskerade decides what client data leaves a penetration tester's machine. A bug in it can send client data to a third party, so security reports are welcome and taken seriously.

## How to report

Report privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**, or go straight to
<https://github.com/adamectimotei/maskerade-proxy/security/advisories/new>.

Please do not open a public issue for anything in the list below.

**Never include real client data in a report.** Reproduce the problem with invented values of the same shape (`jan.novak@acmebank.cz`, `portal.acmebank.cz`, an obviously fake token).

Useful to include:

- the maskerade version (`maskerade --version`) and the tier (`fast` or `full`)
- the `X-Maskerade-Mode` header from an affected response
- the smallest input that reproduces it, with invented values

## What counts as a vulnerability

- A value the pipeline **detected** reaching `api.anthropic.com` unmasked, or a request forwarded after a detector failed. Either breaks the fail-closed promise.
- A reply restored with the wrong value, or a value from one session appearing in another.
- The proxy reachable from other machines in its documented configuration, or the mask-map store readable by another local user.
- A way to make the proxy send traffic anywhere other than `api.anthropic.com`.
- An API key or OAuth token being logged or written to disk.

## What is not

- **A value the detectors do not recognise.** Detection is not complete, and the README says so. Open a normal issue with an invented example of the same shape. If it is a common, well-defined format, such as a vendor's token prefix, say so: that is a pattern that can be added.
- **Fields forwarded without inspection**, such as the system prompt and tool definitions. They are documented, and every response names them in `X-Maskerade-Unscanned`.

## Supported versions

Only the latest release on PyPI. maskerade is alpha software with one maintainer, so handling is best effort, with no fixed response time.

A fix ships as a new release. The advisory is published once that release is on PyPI, and credits the reporter unless they ask not to be named.
