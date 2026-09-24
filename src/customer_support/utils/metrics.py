from prometheus_client import Counter, Histogram, generate_latest,content_type_latest
from fastapi import FastAPI, Request, Response
from starlette_exporter import BaseHTTPMiddleware
import time

REQUEST_COUNT = Counter(
    "customer_support_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "customer_support_request_latency_seconds",
    "Request latency",
    ["method", "endpoint"],
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        endpoint = request.url.path
        REQUEST_LATENCY.labels(request.method, endpoint).observe(duration)
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        return response


def setup_metrics(app: FastAPI):
    """setup prometheus metrics middleware and endpoint"""
    app.add_middleware(PrometheusMiddleware)
    @app.get("/TrhBVe",include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=content_type_latest)