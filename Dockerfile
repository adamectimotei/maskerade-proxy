# maskerade -- localhost masking proxy for the Anthropic API.
#
#   docker build --target light -t maskerade:0.1.4 .
#   docker build --target ner   -t maskerade:0.1.4-ner .
#
# Run it with docker-compose.yml, or by hand:
#
#   docker volume create maskerade-state
#   docker run -d --name maskerade -p 127.0.0.1:8787:8787 \
#       -v maskerade-state:/var/lib/maskerade ghcr.io/adamectimotei/maskerade:0.1.4
#
# THE BIND, READ THIS ONCE
# ------------------------
# The proxy refuses any bind but loopback unless --allow-remote is given.
# Inside a container, a loopback bind is unreachable from the host even with
# -p, so the image binds 0.0.0.0 WITH --allow-remote. The loopback guarantee
# then comes from the publish, `-p 127.0.0.1:8787:8787`, not from the bind.
# Publish without the `127.0.0.1:` and this is an unauthenticated service
# that unmasks client data, open to your whole network. `--network host` would
# keep a real loopback bind, but it only works on Linux, not on Docker
# Desktop for Windows or macOS.
#
# There is deliberately NO `EXPOSE`: `docker run -P` publishes every exposed
# port on ALL interfaces, which is the one mistake this image must not make
# easy.
#
# WHAT IS NOT IN THE IMAGE
# ------------------------
# The BPE encoder (cl100k_base, ~1.6 MB). It has no redistribution licence of
# its own (maskerade/warm.py), so it is fetched on first start into the
# state volume and verified against its published sha256. The light image
# therefore needs network once; every later start is offline.
#
# The NER image bakes in the GLiNER model (Apache-2.0) but fetches the
# encoder the same way. So it starts offline from the SECOND start onward,
# not the first.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# --------------------------------------------------------------- wheel --
# Builds the wheel from the build context. Nothing from this stage ships:
# the runtime stages mount /dist read-only and install from it.
FROM ${PYTHON_IMAGE} AS wheel
WORKDIR /src
COPY pyproject.toml LICENSE ./
COPY maskerade ./maskerade
RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /dist .

# ---------------------------------------------------------------- base --
FROM ${PYTHON_IMAGE} AS base

ARG VERSION=0.1.4
LABEL org.opencontainers.image.title="maskerade" \
      org.opencontainers.image.description="Local masking proxy for the Anthropic API: client data is replaced with placeholders before a request leaves the machine, and restored in the reply." \
      org.opencontainers.image.source="https://github.com/adamectimotei/maskerade-proxy" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}"

# MASKERADE_STATE_DIR holds the mask maps, which are client data. It must be
# a NAMED volume. Docker copies this directory's owner and mode (10001, 0700)
# into an empty named volume on first mount, and the proxy refuses to start
# if it cannot restrict the directory. A bind mount does not get that copy.
#
# HOME and XDG_CACHE_HOME point at /tmp so that any library writing a cache
# writes it to the tmpfs that compose mounts, and not to a read-only root.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MASKERADE_STATE_DIR=/var/lib/maskerade \
    TIKTOKEN_CACHE_DIR=/var/lib/maskerade/tiktoken \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache

RUN groupadd --system --gid 10001 maskerade \
 && useradd --system --uid 10001 --gid 10001 --no-create-home \
            --home-dir /nonexistent --shell /usr/sbin/nologin maskerade \
 && install -d -o 10001 -g 10001 -m 0700 /var/lib/maskerade \
 && install -d -o 10001 -g 10001 -m 0755 /engagement

COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/maskerade-entrypoint

# The proxy looks for ./.maskerade/rules.txt in its working directory, which
# is what `maskerade init` writes. compose mounts the engagement's
# .maskerade/ here read-only, so the rules file is found without a flag.
WORKDIR /engagement

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/healthz', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["/usr/local/bin/maskerade-entrypoint"]

# --------------------------------------------------------------- light --
# custom + regex + form(bpe) + keyval + checksum + id. No torch.
FROM base AS light
RUN --mount=type=bind,from=wheel,source=/dist,target=/dist \
    python -m pip install --no-cache-dir /dist/*.whl
USER 10001:10001
CMD ["proxy", "--host", "0.0.0.0", "--allow-remote", "--port", "8787"]

# ----------------------------------------------------------------- ner --
# The full tier (C2): adds gliner + org_evidence, and the model baked in.
FROM base AS ner

# CPU torch FIRST, from PyTorch's CPU index. The `ner` extra cannot carry an
# index URL, so installing the extra alone pulls the CUDA build (about 2.5 GB
# of libraries a CPU-only tool cannot use). With torch already present, the
# extra resolves against it.
RUN --mount=type=bind,from=wheel,source=/dist,target=/dist \
    python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch \
 && whl="$(ls /dist/*.whl)" \
 && python -m pip install --no-cache-dir "${whl}[ner]"

# Bake the model. The encoder is fetched into a throwaway directory and
# deleted in the SAME layer, so it never ends up in the image (see above).
# The model directory stays root-owned and read-only to the app user.
ENV HF_HOME=/opt/hf
RUN TIKTOKEN_CACHE_DIR=/tmp/bpe-build python -m maskerade.warm --full \
 && rm -rf /tmp/bpe-build /tmp/.cache \
 && chmod -R a+rX /opt/hf

# Set only AFTER the download above, which warm.py refuses to do offline.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

USER 10001:10001
CMD ["proxy", "--host", "0.0.0.0", "--allow-remote", "--port", "8787", "--tier", "full"]
