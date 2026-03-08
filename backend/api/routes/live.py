"""
PlantIQ — Live WebSocket Routes
=================================
WebSocket channels for real-time batch monitoring and alert lifecycle updates.

Endpoints:
  WS /live/{batch_id}        — Primary real-time channel (README-aligned)
  WS /ws/alerts/{batch_id}   — Backward-compatible alias for alert streaming

Message types accepted from client:
  {"type":"ping"}
  {"type":"list_alerts"}
  {
    "type":"transition_alert",
    "alert_id":"ALERT_...",
    "new_state":"delivered|seen|acknowledged|acted_upon|resolved",
    "actor":"operator_1",
    "action_taken":"followed|declined|escalated",
    "action_note":"optional"
  }
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from database import SessionLocal
from database import alert_store

router = APIRouter(tags=["live"])


class LiveConnectionManager:
    """Tracks active websocket clients per batch id for targeted broadcasts."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, batch_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[batch_id].append(websocket)

    def disconnect(self, batch_id: str, websocket: WebSocket) -> None:
        clients = self._connections.get(batch_id, [])
        if websocket in clients:
            clients.remove(websocket)
        if not clients and batch_id in self._connections:
            del self._connections[batch_id]

    async def send_json(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload))

    async def broadcast(self, batch_id: str, payload: dict[str, Any]) -> None:
        clients = list(self._connections.get(batch_id, []))
        for client in clients:
            try:
                await self.send_json(client, payload)
            except Exception:
                # Drop dead sockets so future broadcasts remain healthy.
                self.disconnect(batch_id, client)


manager = LiveConnectionManager()


def _list_batch_alerts(batch_id: str) -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        alerts = alert_store.get_alerts_for_batch(db, batch_id)
        return [a.to_dict() for a in alerts]
    finally:
        db.close()


def _mark_initial_delivery(batch_id: str) -> None:
    """Move newly-fired alerts to delivered once websocket client connects."""
    db = SessionLocal()
    try:
        alerts = alert_store.get_alerts_for_batch(db, batch_id)
        for alert in alerts:
            if alert.state == "fired":
                try:
                    alert_store.transition_alert(db, alert.alert_id, "delivered", actor="system_ws")
                except ValueError:
                    # Ignore race transitions from other clients/processes.
                    pass
    finally:
        db.close()


def _transition_alert(message: dict[str, Any]) -> dict[str, Any]:
    db = SessionLocal()
    try:
        alert = alert_store.transition_alert(
            db,
            str(message.get("alert_id", "")),
            str(message.get("new_state", "")),
            actor=str(message.get("actor", "")) or None,
            action_taken=str(message.get("action_taken", "")) or None,
            action_note=str(message.get("action_note", "")) or None,
        )
        return {"ok": True, "alert": alert.to_dict()}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        db.close()


async def _handle_live_socket(websocket: WebSocket, batch_id: str) -> None:
    await manager.connect(batch_id, websocket)

    # Mark fired alerts as delivered when at least one client is connected.
    _mark_initial_delivery(batch_id)

    await manager.send_json(
        websocket,
        {
            "type": "connected",
            "batch_id": batch_id,
            "alerts": _list_batch_alerts(batch_id),
        },
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(websocket, {"type": "error", "error": "Invalid JSON payload"})
                continue

            msg_type = str(message.get("type", "")).strip().lower()

            if msg_type == "ping":
                await manager.send_json(websocket, {"type": "pong"})
                continue

            if msg_type == "list_alerts":
                await manager.send_json(
                    websocket,
                    {"type": "alerts", "batch_id": batch_id, "alerts": _list_batch_alerts(batch_id)},
                )
                continue

            if msg_type == "transition_alert":
                result = _transition_alert(message)
                if not result["ok"]:
                    await manager.send_json(websocket, {"type": "error", "error": result["error"]})
                    continue

                await manager.broadcast(
                    batch_id,
                    {
                        "type": "alert_transitioned",
                        "batch_id": batch_id,
                        "alert": result["alert"],
                        "alerts": _list_batch_alerts(batch_id),
                    },
                )
                continue

            await manager.send_json(websocket, {"type": "error", "error": f"Unsupported type: {msg_type}"})

    except WebSocketDisconnect:
        manager.disconnect(batch_id, websocket)


@router.websocket("/live/{batch_id}")
async def live_batch_socket(websocket: WebSocket, batch_id: str) -> None:
    await _handle_live_socket(websocket, batch_id)


@router.websocket("/ws/alerts/{batch_id}")
async def live_alert_socket_alias(websocket: WebSocket, batch_id: str) -> None:
    await _handle_live_socket(websocket, batch_id)
