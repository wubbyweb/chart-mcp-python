"""SSE server variant for Chart MCP Python with MCP protocol over SSE."""

import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .server.chart_service import chart_service
from .server.storage import storage
from .shared.schema import ChartConfig, InsertChartRequest, InsertSseEvent


app = FastAPI(
    title="Chart-IMG MCP SSE Server",
    description="Server-Sent Events transport for Chart-IMG MCP server with MCP protocol support and real-time updates",
    version="1.0.0",
)


class GenerateChartRequest(BaseModel):
    """Request model for chart generation."""
    symbol: str
    interval: str
    chartType: str
    duration: str = "1M"
    width: int = 800
    height: int = 600
    theme: str = "light"
    indicators: List[Dict[str, Any]] = []
    drawings: List[Dict[str, Any]] = []
    showVolume: bool = True
    showGrid: bool = True
    timezone: str = "America/New_York"


class ChartResponse(BaseModel):
    """Response model for chart generation."""
    success: bool
    requestId: str
    message: str


# Global dictionary to store SSE connections
connections: Dict[str, asyncio.Queue] = {}


async def event_publisher(request_id: str, event_type: str, message: str) -> None:
    """Publish an event to SSE clients."""
    try:
        # Store event in database
        await storage.create_sse_event(InsertSseEvent(
            requestId=request_id,
            eventType=event_type,
            message=message,
        ))
        
        # Send to connected clients
        for session_id, queue in connections.items():
            try:
                await queue.put({
                    "id": request_id,
                    "event": event_type,
                    "data": json.dumps({
                        "requestId": request_id,
                        "eventType": event_type,
                        "message": message,
                        "timestamp": datetime.now().isoformat(),
                    })
                })
            except Exception as e:
                print(f"Error sending SSE event to {session_id}: {e}", file=sys.stderr)
    
    except Exception as e:
        print(f"Error publishing event: {e}", file=sys.stderr)


async def generate_sse_events(session_id: str):
    """Generate SSE events for a client."""
    # Create a queue for this connection
    queue = asyncio.Queue()
    connections[session_id] = queue
    
    try:
        while True:
            # Wait for events
            event = await queue.get()
            
            # Format as SSE
            sse_data = f"id: {event['id']}\n"
            sse_data += f"event: {event['event']}\n"
            sse_data += f"data: {event['data']}\n\n"
            
            yield sse_data
            
    except asyncio.CancelledError:
        # Client disconnected
        pass
    finally:
        # Clean up connection
        if session_id in connections:
            del connections[session_id]


@app.get("/sse")
async def sse_endpoint(session_id: str = "default"):
    """SSE endpoint for real-time updates."""
    return StreamingResponse(
        generate_sse_events(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    is_configured = chart_service.is_configured()
    
    return {
        "status": "healthy" if is_configured else "configuration_error",
        "chart_api": "configured" if is_configured else "api_key_missing",
        "sse_connections": len(connections),
        "timestamp": datetime.now().isoformat(),
        "message": "All systems operational" if is_configured else "Please set CHART_IMG_API_KEY environment variable",
    }


@app.post("/initialize_chart", response_model=ChartResponse)
async def initialize_chart(request: GenerateChartRequest):
    """Generate a chart with real-time SSE updates."""
    try:
        # Validate chart configuration
        config = ChartConfig(
            symbol=request.symbol,
            interval=request.interval,
            chartType=request.chartType,
            duration=request.duration,
            width=request.width,
            height=request.height,
            theme=request.theme,
            indicators=[],
            drawings=[],
            showVolume=request.showVolume,
            showGrid=request.showGrid,
            timezone=request.timezone,
        )

        # Create chart request
        insert_request = InsertChartRequest(
            symbol=config.symbol,
            interval=config.interval,
            chartType=config.chartType,
            duration=config.duration,
            width=config.width,
            height=config.height,
            indicators=request.indicators,
            drawings=request.drawings,
            theme=config.theme,
            showVolume=config.showVolume,
            showGrid=config.showGrid,
            timezone=config.timezone,
        )
        
        chart_request = await storage.create_chart_request(insert_request)
        request_id = chart_request.requestId

        # Send initial event
        await event_publisher(request_id, "chart_started", f"Starting chart initialization for {config.symbol}")

        # Initialize the chart asynchronously
        asyncio.create_task(process_chart_initialization(config, request_id))

        return ChartResponse(
            success=True,
            requestId=request_id,
            message=f"Chart initialization started for {config.symbol}. Listen to SSE for updates.",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting chart initialization: {str(e)}")


async def process_chart_initialization(config: ChartConfig, request_id: str):
    """Process chart initialization with SSE updates."""
    try:
        # Send progress update
        await event_publisher(request_id, "chart_processing", "Processing chart configuration...")
        
        # Initialize the chart
        result = await chart_service.initialize_chart(config, request_id)

        if result.success:
            await storage.update_chart_request(request_id, {
                "status": "completed",
                "chartUrl": result.url,
                "base64Data": result.base64,
                "processingTime": result.processingTime,
                "completedAt": datetime.now(),
            })

            # Send completion event
            await event_publisher(
                request_id, 
                "chart_completed", 
                f"Chart initialization completed successfully in {(result.processingTime or 0) / 1000:.1f}s"
            )

            # Send chart data event
            if result.base64:
                await event_publisher(
                    request_id,
                    "chart_data",
                    json.dumps({
                        "base64": result.base64,
                        "url": result.url,
                    })
                )

        else:
            await storage.update_chart_request(request_id, {
                "status": "failed",
                "errorMessage": result.error,
                "processingTime": result.processingTime,
                "completedAt": datetime.now(),
            })

            # Send error event
            await event_publisher(request_id, "chart_error", f"Chart initialization failed: {result.error}")

    except Exception as e:
        # Send error event
        await event_publisher(request_id, "chart_error", f"Unexpected error: {str(e)}")


@app.get("/chart_status/{request_id}")
async def get_chart_status(request_id: str):
    """Get chart generation status."""
    try:
        chart_request = await storage.get_chart_request(request_id)
        
        if not chart_request:
            raise HTTPException(status_code=404, detail="Chart request not found")

        response_data = {
            "requestId": chart_request.requestId,
            "symbol": chart_request.symbol,
            "interval": chart_request.interval,
            "chartType": chart_request.chartType,
            "duration": chart_request.duration,
            "status": chart_request.status,
            "createdAt": chart_request.createdAt.isoformat(),
        }

        if chart_request.completedAt:
            response_data["completedAt"] = chart_request.completedAt.isoformat()
        
        if chart_request.processingTime:
            response_data["processingTime"] = chart_request.processingTime / 1000
        
        if chart_request.errorMessage:
            response_data["errorMessage"] = chart_request.errorMessage

        if chart_request.status == "completed" and chart_request.base64Data:
            response_data["base64Data"] = chart_request.base64Data

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chart status: {str(e)}")


@app.get("/events/{request_id}")
async def get_chart_events(request_id: str):
    """Get all SSE events for a chart request."""
    try:
        events = await storage.get_sse_events(request_id)
        
        return {
            "requestId": request_id,
            "events": [
                {
                    "id": event.id,
                    "eventType": event.eventType,
                    "message": event.message,
                    "timestamp": event.timestamp.isoformat(),
                }
                for event in events
            ],
            "count": len(events),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting events: {str(e)}")


@app.get("/symbols")
async def get_available_symbols():
    """Get available trading symbols."""
    try:
        symbols = await chart_service.get_available_symbols()
        
        return {
            "symbols": symbols,
            "total": len(symbols),
            "message": "Use symbols in EXCHANGE:SYMBOL format (e.g., NASDAQ:AAPL, NYSE:TSLA, BINANCE:BTCUSDT)",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting symbols: {str(e)}")


@app.get("/recent_requests")
async def get_recent_requests(limit: int = 10):
    """Get recent chart initialization requests."""
    try:
        requests = await storage.get_recent_chart_requests(limit)

        return {
            "requests": [
                {
                    "requestId": req.requestId,
                    "symbol": req.symbol,
                    "interval": req.interval,
                    "chartType": req.chartType,
                    "duration": req.duration,
                    "status": req.status,
                    "createdAt": req.createdAt.isoformat(),
                }
                for req in requests
            ],
            "count": len(requests),
            "limit": limit,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent requests: {str(e)}")


# Global dictionary to store MCP SSE connections
mcp_connections: Dict[str, asyncio.Queue] = {}


# MCP tool implementation functions (reuse from HTTP server)
async def call_initialize_chart(arguments: Dict[str, Any]) -> str:
    """Initialize chart MCP tool implementation."""
    try:
        # Validate the chart configuration
        config = ChartConfig(
            symbol=arguments.get("symbol"),
            interval=arguments.get("interval"),
            chartType=arguments.get("chartType"),
            duration=arguments.get("duration", "1M"),
            width=arguments.get("width", 800),
            height=arguments.get("height", 600),
            theme=arguments.get("theme", "light"),
            indicators=[],
            drawings=[],
            showVolume=arguments.get("showVolume", True),
            showGrid=arguments.get("showGrid", True),
            timezone=arguments.get("timezone", "America/New_York"),
        )

        # Create chart request
        insert_request = InsertChartRequest(
            symbol=config.symbol,
            interval=config.interval,
            chartType=config.chartType,
            duration=config.duration,
            width=config.width,
            height=config.height,
            indicators=arguments.get("indicators", []),
            drawings=arguments.get("drawings", []),
            theme=config.theme,
            showVolume=config.showVolume,
            showGrid=config.showGrid,
            timezone=config.timezone,
        )
        
        chart_request = await storage.create_chart_request(insert_request)

        # Initialize the chart
        result = await chart_service.initialize_chart(config, chart_request.requestId)

        # Update the chart request with the result
        if result.success:
            await storage.update_chart_request(chart_request.requestId, {
                "status": "completed",
                "chartUrl": result.url,
                "base64Data": result.base64,
                "processingTime": result.processingTime,
                "completedAt": datetime.now(),
            })

            response_text = (
                f"Chart initialized successfully!\n\n"
                f"Request ID: {chart_request.requestId}\n"
                f"Symbol: {config.symbol}\n"
                f"Interval: {config.interval}\n"
                f"Chart Type: {config.chartType}\n"
                f"Duration: {config.duration}\n"
                f"Processing Time: {(result.processingTime or 0) / 1000:.1f}s\n\n"
                f"The chart has been initialized and is available as a PNG image."
            )

            if result.base64:
                # Return with base64 image data
                return f"{response_text}\n[Base64 Image Data Available]"
            
            return response_text

        else:
            await storage.update_chart_request(chart_request.requestId, {
                "status": "failed",
                "errorMessage": result.error,
                "processingTime": result.processingTime,
                "completedAt": datetime.now(),
            })

            raise Exception(f"Chart initialization failed: {result.error}")

    except Exception as e:
        return f"Error initializing chart: {str(e)}"


async def call_get_chart_status(arguments: Dict[str, Any]) -> str:
    """Get chart status MCP tool implementation."""
    try:
        request_id = arguments.get("requestId")
        chart_request = await storage.get_chart_request(request_id)
        
        if not chart_request:
            return "Chart request not found"

        status_parts = [
            "Chart Status Report",
            "",
            f"Request ID: {chart_request.requestId}",
            f"Symbol: {chart_request.symbol}",
            f"Interval: {chart_request.interval}",
            f"Chart Type: {chart_request.chartType}",
            f"Duration: {chart_request.duration}",
            f"Status: {chart_request.status.upper()}",
            f"Created: {chart_request.createdAt.isoformat()}",
        ]

        if chart_request.completedAt:
            status_parts.append(f"Completed: {chart_request.completedAt.isoformat()}")
        
        if chart_request.processingTime:
            status_parts.append(f"Processing Time: {chart_request.processingTime / 1000:.1f}s")
        
        if chart_request.errorMessage:
            status_parts.append(f"Error: {chart_request.errorMessage}")

        if chart_request.status == "completed" and chart_request.base64Data:
            status_parts.append("[Chart image data available]")

        return "\n".join(status_parts)

    except Exception as e:
        return f"Error getting chart status: {str(e)}"


async def call_get_available_symbols() -> str:
    """Get available symbols MCP tool implementation."""
    try:
        symbols = await chart_service.get_available_symbols()
        
        return (
            f"Available Trading Symbols\n\n"
            f"Total symbols available: {len(symbols)}\n\n"
            f"Note: Use symbols in EXCHANGE:SYMBOL format (e.g., NASDAQ:AAPL, NYSE:TSLA, BINANCE:BTCUSDT)\n\n"
            f"Common exchanges:\n"
            f"- NASDAQ: US tech stocks\n"
            f"- NYSE: US stocks\n"
            f"- BINANCE: Crypto pairs\n"
            f"- FOREX: Currency pairs\n"
            f"- CRYPTO: Cryptocurrency symbols"
        )

    except Exception as e:
        return f"Error getting available symbols: {str(e)}"


async def call_get_recent_requests(arguments: Dict[str, Any]) -> str:
    """Get recent requests MCP tool implementation."""
    try:
        limit = arguments.get("limit", 10)
        requests = await storage.get_recent_chart_requests(limit)

        if not requests:
            return "No recent requests found."

        lines = [
            f"Recent Chart Requests ({len(requests)} of {limit})",
            "",
            "Format: Request ID | Symbol | Interval | Type | Duration | Status | Created",
            "",
        ]

        for req in requests:
            line = (
                f"{req.requestId} | {req.symbol} | {req.interval} | "
                f"{req.chartType} | {req.duration} | {req.status.upper()} | "
                f"{req.createdAt.isoformat()}"
            )
            lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting recent requests: {str(e)}"


async def call_health_check() -> str:
    """Health check MCP tool implementation."""
    try:
        is_configured = chart_service.is_configured()
        
        status_parts = [
            "Chart Service Health Check",
            "",
            f"Service Status: {'HEALTHY' if is_configured else 'CONFIGURATION_ERROR'}",
            f"Chart-IMG API: {'CONFIGURED' if is_configured else 'API_KEY_MISSING'}",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
        ]

        if not is_configured:
            status_parts.append(
                "Please set the CHART_IMG_API_KEY environment variable to use the chart generation service."
            )
        else:
            status_parts.append("All systems operational. Ready to initialize charts.")

        return "\n".join(status_parts)

    except Exception as e:
        return f"Error performing health check: {str(e)}"


async def generate_mcp_sse_events(session_id: str):
    """Generate MCP SSE events for a client."""
    # Create a queue for this MCP connection
    queue = asyncio.Queue()
    mcp_connections[session_id] = queue
    
    try:
        while True:
            # Wait for MCP events
            event = await queue.get()
            
            # Format as SSE with MCP JSON-RPC response
            sse_data = f"data: {json.dumps(event)}\n\n"
            
            yield sse_data
            
    except asyncio.CancelledError:
        # Client disconnected
        pass
    finally:
        # Clean up connection
        if session_id in mcp_connections:
            del mcp_connections[session_id]


@app.get("/sse")
async def mcp_sse_endpoint(session_id: str = "default"):
    """MCP protocol over SSE endpoint for @modelcontextprotocol/inspector compatibility."""
    return StreamingResponse(
        generate_mcp_sse_events(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )


@app.post("/sse")
async def mcp_sse_message_endpoint(request: Request, session_id: str = "default"):
    """Handle MCP messages sent to SSE endpoint."""
    try:
        # Get the JSON body
        body = await request.json()
        
        # Process MCP request
        if body.get("method") == "tools/list":
            # Return available tools
            tools = [
                {
                    "name": "initialize_chart",
                    "description": "Initialize TradingView charts using Chart-IMG API with real-time progress updates.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Trading symbol in format EXCHANGE:SYMBOL (e.g., NASDAQ:AAPL)"},
                            "interval": {"type": "string", "description": "Chart time interval"},
                            "chartType": {"type": "string", "description": "Type of chart to generate"},
                            "duration": {"type": "string", "default": "1M", "description": "Chart duration/timeframe"},
                            "width": {"type": "integer", "default": 800, "description": "Chart width in pixels"},
                            "height": {"type": "integer", "default": 600, "description": "Chart height in pixels"},
                            "theme": {"type": "string", "default": "light", "description": "Chart theme"},
                            "indicators": {"type": "array", "default": [], "description": "Technical indicators"},
                            "drawings": {"type": "array", "default": [], "description": "Chart drawings"},
                            "showVolume": {"type": "boolean", "default": True, "description": "Show volume indicator"},
                            "showGrid": {"type": "boolean", "default": True, "description": "Show chart grid"},
                            "timezone": {"type": "string", "default": "America/New_York", "description": "Chart timezone"}
                        },
                        "required": ["symbol", "interval", "chartType"]
                    }
                },
                {
                    "name": "get_chart_status",
                    "description": "Check the status of a chart generation request.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "requestId": {"type": "string", "description": "The request ID returned from generate_chart"}
                        },
                        "required": ["requestId"]
                    }
                },
                {
                    "name": "get_available_symbols",
                    "description": "Get list of available trading symbols from Chart-IMG API.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "get_recent_requests",
                    "description": "Get recent chart initialization requests with their status.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "default": 10, "description": "Maximum number of requests to return"}
                        }
                    }
                },
                {
                    "name": "health_check",
                    "description": "Check the health and configuration status of the chart service.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
            
            response = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "tools": tools
                }
            }
        
        elif body.get("method") == "tools/call":
            # Call a specific tool
            params = body.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "initialize_chart":
                result = await call_initialize_chart(arguments)
            elif tool_name == "get_chart_status":
                result = await call_get_chart_status(arguments)
            elif tool_name == "get_available_symbols":
                result = await call_get_available_symbols()
            elif tool_name == "get_recent_requests":
                result = await call_get_recent_requests(arguments)
            elif tool_name == "health_check":
                result = await call_health_check()
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {tool_name}"
                    }
                }
                # Send error response via SSE
                if session_id in mcp_connections:
                    await mcp_connections[session_id].put(response)
                return {"status": "sent"}
            
            response = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
            }
        
        elif body.get("method") == "initialize":
            # Initialize the MCP session
            response = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "chart-mcp-python",
                        "version": "1.0.0"
                    }
                }
            }
        
        else:
            response = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {body.get('method')}"
                }
            }
        
        # Send response via SSE
        if session_id in mcp_connections:
            await mcp_connections[session_id].put(response)
        
        return {"status": "sent"}
    
    except Exception as e:
        error_response = {
            "jsonrpc": "2.0",
            "id": body.get("id") if "body" in locals() else None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }
        
        # Send error response via SSE
        if session_id in mcp_connections:
            await mcp_connections[session_id].put(error_response)
        
        return {"status": "error", "message": str(e)}


# Keep the original SSE endpoint for progress updates
@app.get("/sse/progress")
async def sse_progress_endpoint(session_id: str = "default"):
    """Original SSE endpoint for real-time chart generation progress updates."""
    return StreamingResponse(
        generate_sse_events(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )


def main() -> None:
    """Run the SSE server."""
    try:
        print("Starting Chart-IMG MCP SSE Server on http://localhost:3002", file=sys.stderr)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=3002,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nShutting down Chart-IMG MCP SSE Server", file=sys.stderr)
    except Exception as e:
        print(f"SSE Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()