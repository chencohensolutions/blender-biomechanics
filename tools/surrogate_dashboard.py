"""Local web page that shows the posture policy network while it trains (no bpy).

Serves tools/surrogate_dashboard.html and the trainer's live snapshot
(`surrogate_live.json`, written by tools/train_posture_surrogate.py) from the
chain package directory. Read-only; nothing is written.

    python tools/surrogate_dashboard.py --package <dir> [--port 8765]

Then open http://localhost:8765 while `train_posture_surrogate.py iterate` runs.
"""

import argparse
import http.server
import os
from pathlib import Path

PAGE = Path(__file__).with_name("surrogate_dashboard.html")
LIVE_FILE = "surrogate_live.json"


def make_handler(package):
    live_path = os.path.join(package, LIVE_FILE)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/live.json"):
                if not os.path.exists(live_path):
                    return self._send(200, b'{"phase": "waiting"}', "application/json")
                with open(live_path, "rb") as handle:
                    return self._send(200, handle.read(), "application/json")
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            self._send(404, b"not found", "text/plain")

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.package))
    print(f"surrogate dashboard: http://localhost:{args.port}  (package {args.package})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
