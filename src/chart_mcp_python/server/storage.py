"""In-memory storage service for Chart MCP Python."""

import time
from datetime import datetime
from typing import Dict, List, Optional, Protocol

from ..shared.schema import (
    ChartRequest,
    InsertChartRequest,
    InsertSseEvent,
    InsertUser,
    SseEvent,
    User,
)


class IStorage(Protocol):
    """Storage interface protocol."""

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        ...

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        ...

    async def create_user(self, user: InsertUser) -> User:
        """Create a new user."""
        ...

    async def create_chart_request(self, request: InsertChartRequest) -> ChartRequest:
        """Create a new chart request."""
        ...

    async def get_chart_request(self, request_id: str) -> Optional[ChartRequest]:
        """Get chart request by ID."""
        ...

    async def update_chart_request(
        self, request_id: str, updates: Dict[str, any]
    ) -> Optional[ChartRequest]:
        """Update chart request."""
        ...

    async def get_recent_chart_requests(self, limit: int) -> List[ChartRequest]:
        """Get recent chart requests."""
        ...

    async def create_sse_event(self, event: InsertSseEvent) -> SseEvent:
        """Create SSE event."""
        ...

    async def get_sse_events(self, request_id: str) -> List[SseEvent]:
        """Get SSE events for request."""
        ...


class MemStorage:
    """In-memory storage implementation."""

    def __init__(self) -> None:
        """Initialize storage."""
        self.users: Dict[int, User] = {}
        self.chart_requests: Dict[str, ChartRequest] = {}
        self.sse_events: Dict[str, List[SseEvent]] = {}
        self.current_user_id = 1
        self.current_chart_id = 1
        self.current_event_id = 1

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    async def create_user(self, insert_user: InsertUser) -> User:
        """Create a new user."""
        user_id = self.current_user_id
        self.current_user_id += 1
        
        user = User(
            id=user_id,
            username=insert_user.username,
            password=insert_user.password,
        )
        self.users[user_id] = user
        return user

    async def create_chart_request(self, insert_request: InsertChartRequest) -> ChartRequest:
        """Create a new chart request."""
        chart_id = self.current_chart_id
        self.current_chart_id += 1
        
        request_id = f"req_{int(time.time())}_{chart_id}"
        now = datetime.now()
        
        request = ChartRequest(
            id=chart_id,
            requestId=request_id,
            symbol=insert_request.symbol,
            interval=insert_request.interval,
            chartType=insert_request.chartType,
            duration=insert_request.duration,
            width=insert_request.width,
            height=insert_request.height,
            indicators=insert_request.indicators,
            drawings=insert_request.drawings,
            theme=insert_request.theme,
            showVolume=insert_request.showVolume,
            showGrid=insert_request.showGrid,
            timezone=insert_request.timezone,
            status="pending",
            chartUrl=None,
            base64Data=None,
            errorMessage=None,
            processingTime=None,
            createdAt=now,
            completedAt=None,
        )
        
        self.chart_requests[request_id] = request
        return request

    async def get_chart_request(self, request_id: str) -> Optional[ChartRequest]:
        """Get chart request by ID."""
        return self.chart_requests.get(request_id)

    async def update_chart_request(
        self, request_id: str, updates: Dict[str, any]
    ) -> Optional[ChartRequest]:
        """Update chart request."""
        existing = self.chart_requests.get(request_id)
        if not existing:
            return None
        
        # Create updated request with new values
        updated_data = existing.model_dump()
        updated_data.update(updates)
        
        updated = ChartRequest(**updated_data)
        self.chart_requests[request_id] = updated
        return updated

    async def get_recent_chart_requests(self, limit: int) -> List[ChartRequest]:
        """Get recent chart requests."""
        requests = list(self.chart_requests.values())
        requests.sort(key=lambda x: x.createdAt, reverse=True)
        return requests[:limit]

    async def create_sse_event(self, insert_event: InsertSseEvent) -> SseEvent:
        """Create SSE event."""
        event_id = self.current_event_id
        self.current_event_id += 1
        
        event = SseEvent(
            id=event_id,
            requestId=insert_event.requestId,
            eventType=insert_event.eventType,
            message=insert_event.message,
            timestamp=datetime.now(),
        )
        
        if insert_event.requestId not in self.sse_events:
            self.sse_events[insert_event.requestId] = []
        
        self.sse_events[insert_event.requestId].append(event)
        return event

    async def get_sse_events(self, request_id: str) -> List[SseEvent]:
        """Get SSE events for request."""
        return self.sse_events.get(request_id, [])

    async def clear_storage(self) -> None:
        """Clear all storage data."""
        self.users.clear()
        self.chart_requests.clear()
        self.sse_events.clear()
        self.current_user_id = 1
        self.current_chart_id = 1
        self.current_event_id = 1

    async def delete_chart_request(self, request_id: str) -> bool:
        """Delete a specific chart request and its related data."""
        if request_id not in self.chart_requests:
            return False
        
        # Remove chart request
        del self.chart_requests[request_id]
        
        # Remove related SSE events
        if request_id in self.sse_events:
            del self.sse_events[request_id]
        
        return True


# Global storage instance
storage = MemStorage()