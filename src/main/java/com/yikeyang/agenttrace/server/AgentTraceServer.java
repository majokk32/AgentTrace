package com.yikeyang.agenttrace.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import com.yikeyang.agenttrace.model.DeduplicationRequest;
import com.yikeyang.agenttrace.model.SearchByTrajectoryRequest;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.Trajectory;
import com.yikeyang.agenttrace.search.TrajectorySearchBackend;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class AgentTraceServer implements AutoCloseable {

    private final TrajectorySearchBackend backend;
    private final List<Trajectory> trajectories;
    private final Map<String, Trajectory> trajectoriesById;
    private final ObjectMapper objectMapper;
    private final HttpServer server;
    private final ExecutorService executor;
    private final OperationalTelemetry telemetry;

    public AgentTraceServer(
            int port,
            TrajectorySearchBackend backend,
            List<Trajectory> trajectories,
            ObjectMapper objectMapper) throws IOException {
        this.backend = backend;
        this.trajectories = List.copyOf(trajectories);
        this.trajectoriesById = trajectories.stream().collect(
                Collectors.toUnmodifiableMap(Trajectory::id, Function.identity()));
        this.objectMapper = objectMapper;
        this.server = HttpServer.create(new InetSocketAddress(port), 0);
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
        this.telemetry = new OperationalTelemetry();
        server.setExecutor(executor);
        registerRoutes();
    }

    public void start() {
        server.start();
    }

    public int port() {
        return server.getAddress().getPort();
    }

    @Override
    public void close() {
        server.stop(1);
        executor.close();
    }

    private void registerRoutes() {
        server.createContext("/health", exchange -> handle(
                exchange, "GET", "/health", false, ignored -> Map.of("status", "ok")));
        server.createContext("/api/telemetry", exchange -> handle(
                exchange, "GET", "/api/telemetry", false,
                ignored -> telemetry.snapshot(backend.stats())));
        server.createContext("/dashboard", exchange -> handleDashboard(exchange));
        server.createContext("/api/stats", exchange -> handle(
                exchange, "GET", "/api/stats", true, ignored -> backend.stats()));
        server.createContext("/api/search", exchange -> handle(
                exchange, "POST", "/api/search", true, input -> {
                    SearchRequest request = objectMapper.readValue(input, SearchRequest.class);
                    return backend.search(request);
                }));
        server.createContext("/api/search/by-trajectory", exchange -> handle(
                exchange, "POST", "/api/search/by-trajectory", true, input -> {
                    SearchByTrajectoryRequest request =
                            objectMapper.readValue(input, SearchByTrajectoryRequest.class);
                    if (request.trajectoryId() == null || request.trajectoryId().isBlank()) {
                        throw new IllegalArgumentException("trajectoryId is required");
                    }
                    Trajectory source = trajectoriesById.get(request.trajectoryId());
                    if (source == null) {
                        throw new IllegalArgumentException(
                                "unknown trajectoryId: " + request.trajectoryId());
                    }
                    int requestedK = request.requestedK();
                    if (requestedK < 1 || requestedK > 100) {
                        throw new IllegalArgumentException("k must be between 1 and 100");
                    }
                    SearchRequest searchRequest = new SearchRequest(
                            source.embedding(),
                            Math.min(100, requestedK + 1),
                            source.platform(),
                            source.app(),
                            request.success());
                    return backend.search(searchRequest).stream()
                            .filter(result -> !result.id().equals(source.id()))
                            .limit(requestedK)
                            .toList();
                }));
        server.createContext("/api/deduplicate", exchange -> handle(
                exchange, "POST", "/api/deduplicate", true, input -> {
                    DeduplicationRequest request =
                            objectMapper.readValue(input, DeduplicationRequest.class);
                    return backend.findDuplicateGroups(
                            trajectories,
                            request.requestedThreshold(),
                            request.requestedCandidateK());
                }));
    }

    private void handle(
            HttpExchange exchange,
            String method,
            String route,
            boolean observed,
            ExchangeAction action)
            throws IOException {
        OperationalTelemetry.RequestObservation observation = observed
                ? telemetry.start(method + " " + route)
                : null;
        int status = 500;
        Exception failure = null;
        try {
            if (!method.equalsIgnoreCase(exchange.getRequestMethod())) {
                status = 405;
                writeJson(exchange, status, Map.of("error", "method not allowed"));
                return;
            }
            Object response = action.execute(exchange.getRequestBody());
            status = 200;
            writeJson(exchange, status, response);
        } catch (IllegalArgumentException exception) {
            status = 400;
            failure = exception;
            writeJson(exchange, status, Map.of("error", exception.getMessage()));
        } catch (Exception exception) {
            status = 500;
            failure = exception;
            writeJson(exchange, status, Map.of(
                    "error", "internal server error",
                    "detail", exception.getMessage() == null
                            ? exception.getClass().getSimpleName()
                            : exception.getMessage()));
        } finally {
            if (observation != null) {
                observation.complete(status, failure);
            }
            exchange.close();
        }
    }

    private void handleDashboard(HttpExchange exchange) throws IOException {
        try {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                writeJson(exchange, 405, Map.of("error", "method not allowed"));
                return;
            }
            byte[] payload;
            try (InputStream input = AgentTraceServer.class.getResourceAsStream(
                    "/dashboard.html")) {
                if (input == null) {
                    throw new IOException("dashboard resource is missing");
                }
                payload = input.readAllBytes();
            }
            exchange.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
            exchange.getResponseHeaders().set(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'");
            exchange.sendResponseHeaders(200, payload.length);
            try (OutputStream outputStream = exchange.getResponseBody()) {
                outputStream.write(payload);
            }
        } finally {
            exchange.close();
        }
    }

    private void writeJson(HttpExchange exchange, int status, Object body) throws IOException {
        byte[] payload = objectMapper.writeValueAsBytes(body);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, payload.length);
        try (OutputStream outputStream = exchange.getResponseBody()) {
            outputStream.write(payload);
        }
    }

    @FunctionalInterface
    private interface ExchangeAction {
        Object execute(InputStream input) throws Exception;
    }
}
