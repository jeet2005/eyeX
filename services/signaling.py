"""
WebSocket signaling server for WebRTC connections
"""
import asyncio
import json
import logging
from typing import Dict, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import WebSocket

logger = logging.getLogger("EYE_X")

@dataclass
class Connection:
    """Represents a connected client"""
    websocket: WebSocket
    room: str
    connected_at: datetime = field(default_factory=datetime.utcnow)
    client_id: str = ""

    def __hash__(self):
        return hash(self.client_id)

    def __eq__(self, other):
        if not isinstance(other, Connection):
            return False
        return self.client_id == other.client_id
    
    
class ConnectionManager:
    """
    Manages WebSocket connections for WebRTC signaling.
    Supports room-based connections (gate, classroom) for different cameras.
    """
    
    def __init__(self):
        # Room -> Set of connections
        self.rooms: Dict[str, Set[Connection]] = {
            "gate": set(),
            "classroom": set(),
            "dashboard": set()
        }
        # WebSocket -> Connection mapping for easy lookup
        self.connections: Dict[WebSocket, Connection] = {}
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket, room: str, client_id: str = "") -> Connection:
        """Accept a new WebSocket connection and add to room"""
        await websocket.accept()
        
        connection = Connection(
            websocket=websocket,
            room=room,
            client_id=client_id or f"client_{id(websocket)}"
        )
        
        async with self._lock:
            if room not in self.rooms:
                self.rooms[room] = set()
            self.rooms[room].add(connection)
            self.connections[websocket] = connection
        
        logger.info(f"Client {connection.client_id} connected to room '{room}'")
        
        # Notify dashboard about new connection
        await self.broadcast_to_room("dashboard", {
            "type": "camera_connected",
            "room": room,
            "client_id": connection.client_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return connection
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a connection"""
        async with self._lock:
            connection = self.connections.pop(websocket, None)
            if connection:
                self.rooms[connection.room].discard(connection)
                logger.info(f"Client {connection.client_id} disconnected from room '{connection.room}'")
                
                # Notify dashboard
                await self._broadcast_to_room_internal("dashboard", {
                    "type": "camera_disconnected",
                    "room": connection.room,
                    "client_id": connection.client_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self.disconnect(websocket)
    
    async def broadcast_to_room(self, room: str, message: dict, exclude: Optional[WebSocket] = None):
        """Broadcast message to all clients in a room"""
        async with self._lock:
            await self._broadcast_to_room_internal(room, message, exclude)
    
    async def _broadcast_to_room_internal(self, room: str, message: dict, exclude: Optional[WebSocket] = None):
        """Internal broadcast without lock (must be called with lock held)"""
        if room not in self.rooms:
            return
            
        disconnected = []
        for conn in self.rooms[room]:
            if conn.websocket != exclude:
                try:
                    await conn.websocket.send_json(message)
                except Exception:
                    disconnected.append(conn.websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            connection = self.connections.pop(ws, None)
            if connection:
                self.rooms[room].discard(connection)
    
    async def relay_to_dashboard(self, message: dict, from_room: str):
        """Relay camera signaling messages to dashboard"""
        message["from_room"] = from_room
        await self.broadcast_to_room("dashboard", message)
    
    async def relay_to_camera(self, message: dict, to_room: str):
        """Relay dashboard signaling messages to specific camera"""
        await self.broadcast_to_room(to_room, message)
    
    def get_room_count(self, room: str) -> int:
        """Get number of connections in a room"""
        return len(self.rooms.get(room, []))
    
    def get_room_connections(self, room: str) -> list:
        """Get all connections in a room"""
        return list(self.rooms.get(room, []))
    
    def get_all_rooms_status(self) -> dict:
        """Get status of all rooms"""
        return {
            room: {
                "count": len(connections),
                "clients": [c.client_id for c in connections]
            }
            for room, connections in self.rooms.items()
        }


# Global connection manager instance
manager = ConnectionManager()
