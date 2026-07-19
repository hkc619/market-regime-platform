import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger


logger = get_logger("request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()

        

        logger.info(
            "Request started | request_id=%s | method=%s | path=%s | client=%s",
            request_id,
            request.method,
            request.url.path,
            request.client.host if request.client else None,
        )
        

        try:
            response = await call_next(request)

        except Exception:
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.exception(
                "Request failed | request_id=%s | method=%s | path=%s | latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
            )
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request completed | request_id=%s | method=%s | path=%s | status_code=%s | latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        return response