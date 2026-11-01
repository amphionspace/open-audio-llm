"""Forward OpenAI-compatible requests to a vLLM server, replacing one system-prompt string.

AmphionEval sends the training TS instruction ("3-second"); audiollm-go main sends
"5-second" (targetasr/service.go). This proxy lets the v12 suite measure the deployed
string without changing AmphionEval. Every other byte of the request is forwarded as is.
"""
import argparse
import json

from aiohttp import ClientSession, ClientTimeout, web


def make_app(backend, old, new):
    counts = {"rewritten": 0, "unchanged": 0}

    async def forward(request):
        body = await request.read()
        if request.method == "POST" and request.path == "/v1/chat/completions":
            payload = json.loads(body)
            for message in payload.get("messages", []):
                if message.get("role") == "system" and isinstance(message.get("content"), str):
                    if old in message["content"]:
                        message["content"] = message["content"].replace(old, new)
                        counts["rewritten"] += 1
                    else:
                        counts["unchanged"] += 1
            body = json.dumps(payload).encode()
        async with request.app["http"].request(request.method, backend + request.path_qs, data=body,
                                               headers={"Content-Type": request.content_type}) as response:
            return web.Response(body=await response.read(), status=response.status,
                                content_type=response.content_type)

    async def stats(_):
        return web.json_response(counts)

    async def startup(app):
        app["http"] = ClientSession(timeout=ClientTimeout(total=900))

    async def cleanup(app):
        await app["http"].close()

    app = web.Application(client_max_size=64 << 20)
    app.router.add_get("/proxy-stats", stats)
    app.router.add_route("*", "/{tail:.*}", forward)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()
    web.run_app(make_app(args.backend.rstrip("/"), args.old, args.new), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
