# backend/app/ws_hub.py
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import DefaultDict, Set

from fastapi import WebSocket


class WebSocketHub:
    """
    Tracks websocket clients per run_id and allows broadcasting messages.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._clients: DefaultDict[str, Set[WebSocket]] = defaultdict(set)

    async def add(self, run_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._clients[run_id].add(ws)

    async def remove(self, run_id: str, ws: WebSocket) -> None:
        async with self._lock:
            if run_id in self._clients:
                self._clients[run_id].discard(ws)
                if not self._clients[run_id]:
                    del self._clients[run_id]

    async def publish_text(self, run_id: str, text: str) -> None:
        # snapshot list under lock; send outside lock
        async with self._lock:
            targets = list(self._clients.get(run_id, set()))

        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.get(run_id, set()).discard(ws)


HUB = WebSocketHub()
