#!/usr/bin/env bash
# Smoke test for the maskerade images. Run by .github/workflows/docker-smoke.yml.
#
#   bash .github/smoke-test.sh maskerade:ci            light image
#   bash .github/smoke-test.sh --ner maskerade:ci-ner  NER image
#
# Never pushes anything. Uses a MOCK Anthropic upstream inside a private
# Docker network, so no API key is needed and nothing leaves the runner except
# the one-time encoder download. The light test checks that:
#   - the image runs as uid 10001, the state directory is 10001-owned 0700, and
#     there is no EXPOSE (so `docker run -P` cannot publish to all interfaces)
#   - the first start fetches the encoder into the volume; later starts do not
#   - under the compose hardening (read-only root, no capabilities) the proxy
#     masks planted client values -- including an email cut by a domain rule
#     and a %40-encoded email -- and none of them reaches the upstream
#   - the reply is unmasked for the client
#   - a named volume keeps the session across restart AND container
#     replacement (X-Maskerade-Session: continued, same placeholders)
#   - -p 127.0.0.1:... listens on loopback only
#   - a state directory the app user does not own is refused, not used
set -euo pipefail
# No `cmd | grep -q` anywhere below. Under pipefail, grep -q exits on the first
# match, the writer gets SIGPIPE, and the pipeline FAILS although the text was
# there -- a flaky check, and inverted on the `&& fail` checks. Logs are read
# into a variable once and matched with [[ ... ]].

NER=0
if [ "${1:-}" = "--ner" ]; then NER=1; shift; fi
IMG="${1:?usage: smoke-test.sh [--ner] IMAGE}"
PORT=18787
WORK="$(mktemp -d)"

cleanup() {
    docker rm -f mk-px mk-mock mk-ner >/dev/null 2>&1 || true
    docker network rm mk-net >/dev/null 2>&1 || true
    docker volume rm -f mk-state mk-ner-state >/dev/null 2>&1 || true
    docker image rm -f mk-smoke-mockup >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

ok()   { echo "ok    $*"; }
fail() { echo "FAIL  $*"; echo "---- proxy log ----"; docker logs mk-px 2>&1 | tail -40 || true; exit 1; }

# ------------------------------------------------------------------ NER --
if [ "$NER" = 1 ]; then
    docker volume create mk-ner-state >/dev/null
    # The encoder is fetched once, WITH network (see the Dockerfile).
    docker run --rm -v mk-ner-state:/var/lib/maskerade --entrypoint python "$IMG" -m maskerade.warm >/dev/null
    ok "encoder fetched into the volume"
    docker run -d --name mk-ner --network none -v mk-ner-state:/var/lib/maskerade "$IMG" >/dev/null
    up=""
    for _ in $(seq 1 120); do
        if up=$(docker exec mk-ner python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8787/healthz',timeout=3).read().decode())" 2>/dev/null); then break; fi
        sleep 5
    done
    [ -n "$up" ] || { docker logs mk-ner 2>&1 | tail -40; echo "FAIL  NER proxy never answered /healthz with no network"; exit 1; }
    ok "NER image serves /healthz with --network none"
    [[ "$up" == *"GLINER"* ]] || { echo "FAIL  mode lacks GLINER: $up"; exit 1; }
    [[ "$up" == *"ORG_EVIDENCE"* ]] || { echo "FAIL  mode lacks ORG_EVIDENCE: $up"; exit 1; }
    ok "mode: $(echo "$up" | python3 -c 'import json,sys;print(json.load(sys.stdin)["mode"])')"
    docker exec mk-ner maskerade --version
    echo "PASSED (ner)"
    exit 0
fi

# ---------------------------------------------------------------- light --
[ "$(docker run --rm --entrypoint id "$IMG" -u)" = "10001" ] || fail "image does not run as uid 10001"
ok "runs as uid 10001"
st="$(docker run --rm --entrypoint stat "$IMG" -c '%u %a' /var/lib/maskerade)"
[ "$st" = "10001 700" ] || fail "state dir is '$st', want '10001 700'"
ok "state dir 10001 0700"
[ "$(docker inspect -f '{{json .Config.ExposedPorts}}' "$IMG")" = "null" ] || fail "image declares EXPOSE"
ok "no EXPOSE"

# A TEST-ONLY derivative pointing the proxy at the mock instead of Anthropic.
printf 'FROM %s\nUSER root\nRUN python -c "%s"\nUSER 10001:10001\n' "$IMG" \
  "import maskerade.proxy.server as s;p=s.__file__;t=open(p).read();n=t.replace('UPSTREAM = \\\"https://api.anthropic.com\\\"','UPSTREAM = \\\"http://mk-mock:9999\\\"',1);assert n!=t;open(p,'w').write(n)" \
  > "$WORK/Dockerfile.mockup"
docker build -q -t mk-smoke-mockup -f "$WORK/Dockerfile.mockup" "$WORK" >/dev/null

cat > "$WORK/mock.py" <<'EOF'
import json, http.server
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        b = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with open("/tmp/received.jsonl", "ab") as f:
            f.write(b + b"\n")
        last = json.loads(b)["messages"][-1]["content"]
        out = json.dumps({"id": "msg_mock", "type": "message", "role": "assistant",
                          "content": [{"type": "text", "text": "ECHO: " + last}],
                          "stop_reason": "end_turn",
                          "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
http.server.ThreadingHTTPServer(("0.0.0.0", 9999), H).serve_forever()
EOF
mkdir -p "$WORK/eng/.maskerade"
printf 'acmebank.cz = [CLIENT_DOMAIN]\nAcme Bank a.s. = [CLIENT_NAME]\n' > "$WORK/eng/.maskerade/rules.txt"
chmod -R a+rX "$WORK"

docker network create mk-net >/dev/null
docker volume create mk-state >/dev/null
docker run -d --name mk-mock --network mk-net -v "$WORK/mock.py:/mock.py:ro" \
    --entrypoint python "$IMG" /mock.py >/dev/null

start_proxy() {
    docker run -d --name mk-px --network mk-net -p "127.0.0.1:$PORT:8787" \
        -v mk-state:/var/lib/maskerade -v "$WORK/eng/.maskerade:/engagement/.maskerade:ro" \
        --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges:true \
        mk-smoke-mockup proxy --host 0.0.0.0 --allow-remote --port 8787 --engagement eng-ci >/dev/null
    for _ in $(seq 1 60); do
        curl -fs -m 3 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && return 0
        sleep 2
    done
    fail "proxy never answered /healthz"
}

start_proxy
logs="$(docker logs mk-px 2>&1)"
[[ "$logs" == *"fetching it once"* ]] || fail "first start did not fetch the encoder"
ok "first start fetched the encoder (checksum-verified by warm)"
[[ "$logs" == *"ALL INTERFACES"* ]] || fail "banner does not report the --allow-remote bind"
[[ "$logs" == *"FORM(bpe)"* ]] || fail "form is not running BPE"
ok "banner: ALL INTERFACES, FORM(bpe), rules loaded"

listen="$(ss -ltnH "sport = :$PORT" | awk '{print $4}' | sort -u | tr '\n' ' ')"
case "$listen" in
    "127.0.0.1:$PORT ") ok "host listens on 127.0.0.1:$PORT only" ;;
    *) fail "host listeners for :$PORT are '$listen', want only 127.0.0.1:$PORT" ;;
esac

python3 - "$WORK" <<'EOF'
import json, sys
w = sys.argv[1]
u1 = {"role": "user", "content": "Client Acme Bank a.s. runs portal.acmebank.cz. Admin is jan.novak@acmebank.cz, reset /reset?email=eva.k%40acmebank.cz, host 10.20.30.40, password=Pentest#123."}
json.dump({"model": "m", "max_tokens": 16, "messages": [u1]}, open(w + "/t1.json", "w"))
EOF
send() {  # send BODYFILE OUTFILE -> prints the X-Maskerade-Session header value
    # No API key header: the mock upstream does not check one, and a fake key
    # in a public repo is still a secret-scanner finding.
    curl -fs -D "$WORK/h" -H "content-type: application/json" \
         -H "anthropic-version: 2023-06-01" --data-binary @"$1" \
         "http://127.0.0.1:$PORT/v1/messages" > "$2" || fail "request $1 failed"
    grep -i '^x-maskerade-session:' "$WORK/h" | cut -d' ' -f2- | tr -d '\r'
}
s1="$(send "$WORK/t1.json" "$WORK/r1.json")"
case "$s1" in "eng-ci; new") ok "turn 1: session eng-ci; new" ;; *) fail "turn 1 session header '$s1'" ;; esac

check_upstream() {
    docker exec mk-mock cat /tmp/received.jsonl > "$WORK/received.jsonl"
    python3 - "$WORK/received.jsonl" <<'EOF' || fail "planted values reached the upstream"
import sys
t = open(sys.argv[1], encoding="utf-8").read().lower()
leaked = [v for v in ("jan.novak", "eva.k", "acme bank", "acmebank", "10.20.30.40", "pentest#123") if v in t]
print("      upstream leaks:", leaked or "none")
sys.exit(1 if leaked else 0)
EOF
}
check_upstream
ok "no planted value reached the upstream (incl. the cut and the %40 email)"

python3 - "$WORK" <<'EOF' || fail "the reply was not unmasked for the client"
import json, sys
r = json.load(open(sys.argv[1] + "/r1.json"))["content"][0]["text"]
sys.exit(0 if "jan.novak@acmebank.cz" in r and "eva.k%40acmebank.cz" in r and "10.20.30.40" in r else 1)
EOF
ok "reply unmasked for the client"

python3 - "$WORK" <<'EOF'
import json, sys
w = sys.argv[1]
t1 = json.load(open(w + "/t1.json")); r1 = json.load(open(w + "/r1.json"))
m = t1["messages"] + [{"role": "assistant", "content": r1["content"][0]["text"]},
                      {"role": "user", "content": "Again jan.novak@acmebank.cz and 10.20.30.40."}]
json.dump({"model": "m", "max_tokens": 16, "messages": m}, open(w + "/t2.json", "w"))
EOF
last_user() { docker exec mk-mock python -c "import json;print(json.loads(open('/tmp/received.jsonl').read().splitlines()[-1])['messages'][-1]['content'])"; }

docker restart mk-px >/dev/null
for _ in $(seq 1 60); do curl -fs -m 3 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break; sleep 2; done
s2="$(send "$WORK/t2.json" "$WORK/r2.json")"; u2="$(last_user)"
[ "$s2" = "eng-ci; continued" ] || fail "after restart, session header '$s2'"
case "$u2" in *"[EMAIL_ADDRESS_1]"*"[INTERNAL_IP_1]"*) ;; *) fail "after restart, placeholders changed: $u2" ;; esac
ok "after docker restart: continued, same placeholders"

docker rm -f mk-px >/dev/null
start_proxy
logs="$(docker logs mk-px 2>&1)"
[[ "$logs" == *"fetching it once"* ]] && fail "replacement container re-fetched the encoder"
s3="$(send "$WORK/t2.json" "$WORK/r3.json")"; u3="$(last_user)"
[ "$s3" = "eng-ci; continued" ] || fail "after replacement, session header '$s3'"
[ "$u3" = "$u2" ] || fail "after replacement, placeholders changed: $u3"
ok "after container replacement: continued, same placeholders, encoder not re-fetched"
check_upstream
ok "still nothing planted upstream after three requests"

# A state directory the app user does not own: refused, not used.
mkdir -p "$WORK/bind" && chmod 755 "$WORK/bind"
if out="$(docker run --rm -v "$WORK/bind:/var/lib/maskerade" "$IMG" sessions list 2>&1)"; then
    fail "sessions list ran on a directory it cannot restrict: $out"
fi
[[ "$out" == *"cannot restrict"* || "$out" == *"ignored the change"* ]] || fail "unexpected refusal text: $out"
ok "bind-mounted state dir refused: $(echo "$out" | grep -o 'cannot restrict[^(]*\|is mode [^:]*' | head -1)"

echo "PASSED (light)"
