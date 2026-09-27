package com.yikeyang.agenttrace.server;

import com.yikeyang.agenttrace.model.IndexStats;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;

final class OperationalTelemetry {

    private static final long SLOW_REQUEST_MILLIS = 1_000;
    private static final long MINIMUM_REQUESTS_FOR_ERROR_ALERT = 20;
    private static final double ERROR_RATE_ALERT_THRESHOLD = 0.05;

    private final Instant startedAt = Instant.now();
    private final long startedNanos = System.nanoTime();
    private final LongAdder totalRequests = new LongAdder();
    private final LongAdder inFlightRequests = new LongAdder();
    private final LongAdder successfulRequests = new LongAdder();
    private final LongAdder clientErrors = new LongAdder();
    private final LongAdder serverErrors = new LongAdder();
    private final LongAdder totalDurationNanos = new LongAdder();
    private final AtomicLong maxDurationNanos = new AtomicLong();
    private final Map<String, RouteMetrics> routes = new ConcurrentHashMap<>();
    private final AtomicReference<FailureSnapshot> lastServerError = new AtomicReference<>();

    RequestObservation start(String route) {
        totalRequests.increment();
        inFlightRequests.increment();
        return new RequestObservation(route, System.nanoTime());
    }

    TelemetrySnapshot snapshot(IndexStats backend) {
        long requests = totalRequests.sum();
        long successes = successfulRequests.sum();
        long clientErrorCount = clientErrors.sum();
        long serverErrorCount = serverErrors.sum();
        long durationNanos = totalDurationNanos.sum();
        double errorRate = requests == 0
                ? 0.0
                : (double) (clientErrorCount + serverErrorCount) / requests;
        double serverErrorRate = requests == 0
                ? 0.0
                : (double) serverErrorCount / requests;

        List<String> alerts = new ArrayList<>();
        if (requests >= MINIMUM_REQUESTS_FOR_ERROR_ALERT
                && serverErrorRate >= ERROR_RATE_ALERT_THRESHOLD) {
            alerts.add("Server error rate is at or above 5%.");
        }
        if (nanosToMillis(maxDurationNanos.get()) >= SLOW_REQUEST_MILLIS) {
            alerts.add("At least one workload request exceeded 1000 ms.");
        }

        Map<String, RouteSnapshot> routeSnapshots = new TreeMap<>();
        routes.forEach((route, metrics) -> routeSnapshots.put(route, metrics.snapshot()));
        return new TelemetrySnapshot(
                Instant.now().toString(),
                startedAt.toString(),
                Duration.ofNanos(System.nanoTime() - startedNanos).toSeconds(),
                alerts.isEmpty() ? "ok" : "degraded",
                backend,
                new WorkloadSnapshot(
                        requests,
                        inFlightRequests.sum(),
                        successes,
                        clientErrorCount,
                        serverErrorCount,
                        errorRate),
                new LatencySnapshot(
                        requests == 0 ? 0.0 : nanosToMillis(durationNanos) / requests,
                        nanosToMillis(maxDurationNanos.get())),
                Map.copyOf(routeSnapshots),
                List.copyOf(alerts),
                lastServerError.get());
    }

    private void complete(String route, long started, int status, Exception exception) {
        long durationNanos = Math.max(0, System.nanoTime() - started);
        inFlightRequests.decrement();
        totalDurationNanos.add(durationNanos);
        maxDurationNanos.accumulateAndGet(durationNanos, Math::max);
        RouteMetrics routeMetrics = routes.computeIfAbsent(route, ignored -> new RouteMetrics());
        routeMetrics.record(status, durationNanos);
        if (status >= 500) {
            serverErrors.increment();
            lastServerError.set(new FailureSnapshot(
                    Instant.now().toString(),
                    route,
                    exception == null ? "unknown" : exception.getClass().getSimpleName(),
                    exception == null || exception.getMessage() == null
                            ? "No diagnostic message available"
                            : exception.getMessage()));
        } else if (status >= 400) {
            clientErrors.increment();
        } else {
            successfulRequests.increment();
        }
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    final class RequestObservation {

        private final String route;
        private final long started;
        private boolean completed;

        private RequestObservation(String route, long started) {
            this.route = route;
            this.started = started;
        }

        void complete(int status, Exception exception) {
            if (completed) {
                return;
            }
            completed = true;
            OperationalTelemetry.this.complete(route, started, status, exception);
        }
    }

    private static final class RouteMetrics {

        private final LongAdder requests = new LongAdder();
        private final LongAdder errors = new LongAdder();
        private final LongAdder totalDurationNanos = new LongAdder();
        private final AtomicLong maxDurationNanos = new AtomicLong();

        void record(int status, long durationNanos) {
            requests.increment();
            if (status >= 400) {
                errors.increment();
            }
            totalDurationNanos.add(durationNanos);
            maxDurationNanos.accumulateAndGet(durationNanos, Math::max);
        }

        RouteSnapshot snapshot() {
            long requestCount = requests.sum();
            long errorCount = errors.sum();
            return new RouteSnapshot(
                    requestCount,
                    errorCount,
                    requestCount == 0 ? 0.0 : (double) errorCount / requestCount,
                    requestCount == 0
                            ? 0.0
                            : nanosToMillis(totalDurationNanos.sum()) / requestCount,
                    nanosToMillis(maxDurationNanos.get()));
        }
    }

    record TelemetrySnapshot(
            String generatedAt,
            String startedAt,
            long uptimeSeconds,
            String status,
            IndexStats backend,
            WorkloadSnapshot workload,
            LatencySnapshot latencyMs,
            Map<String, RouteSnapshot> routes,
            List<String> alerts,
            FailureSnapshot lastServerError) {
    }

    record WorkloadSnapshot(
            long totalRequests,
            long inFlightRequests,
            long successfulRequests,
            long clientErrors,
            long serverErrors,
            double errorRate) {
    }

    record LatencySnapshot(double mean, double max) {
    }

    record RouteSnapshot(
            long requests,
            long errors,
            double errorRate,
            double meanLatencyMs,
            double maxLatencyMs) {
    }

    record FailureSnapshot(
            String occurredAt,
            String route,
            String exceptionType,
            String message) {
    }
}
