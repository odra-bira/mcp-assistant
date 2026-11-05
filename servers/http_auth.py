"""
Bearer-token authentication middleware for remote (HTTP) MCP servers.

Rejects any request whose header is not exactly `Authorization: Bearer <token>`
with 401 before it reaches the MCP application. Comparison is constant-time.

Addresses the authentication gap identified in Hou et al. (2025), Section 6.2.
A shared secret is the minimal form; the MCP specification's OAuth 2.1 flow
(per-user, short-lived tokens) is the production-grade successor.
"""

import hmac
import json


class BearerTokenMiddleware:
    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":  # lifespan events etc. pass straight through
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        given = headers.get(b"authorization", b"")
        # compare_digest takes the same time whether the first or last character is wrong,
        # so attackers can't guess the token one character at a time ("timing attack").
        if not hmac.compare_digest(given, self.expected):
            body = json.dumps({"error": "unauthorized"}).encode()
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
