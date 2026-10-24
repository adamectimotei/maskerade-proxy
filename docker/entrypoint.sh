#!/bin/sh
# maskerade container entrypoint.
#
#   (no args)                 runs CMD: the proxy
#   sessions list             any other maskerade subcommand runs as given
#
# Before `proxy` it makes sure the BPE encoder is in the state volume. The
# encoder is not shipped in the image (no redistribution licence; see
# maskerade/warm.py), so the first start fetches it once (~1.7 MB, checksum
# verified) and every later start is offline. Without network on the first
# start, warm fails and the container exits. That is deliberate: the form
# stage cannot run without the encoder, and the proxy would refuse to start
# anyway.
set -eu

if [ "${1:-}" = "proxy" ]; then
    if ! python -m maskerade.warm --check >/dev/null 2>&1; then
        echo "  [entrypoint] BPE encoder not in the state volume yet; fetching it once." >&2
        if ! python -m maskerade.warm >&2; then
            echo "" >&2
            echo "  [entrypoint] could not fetch the BPE encoder, so the proxy is NOT starting." >&2
            echo "  The first start needs HTTPS to openaipublic.blob.core.windows.net, once." >&2
            echo "  Offline? Run 'python -m maskerade.warm --to DIR' on a connected machine and" >&2
            echo "  copy DIR/tiktoken/* into the state volume's tiktoken/ directory." >&2
            exit 1
        fi
    fi
fi

exec maskerade "$@"
