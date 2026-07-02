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
import java.nio.charset.StandardCharsets;
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
        server.setExecutor(executor);
        registerRoutes();
    }

    public void start() {
        server.start();
    }

    @Override
    public void close() {
        server.stop(1);
        executor.close();
    }

    private void registerRoutes() {
        server.createContext("/health", exchange -> handle(exchange, "GET", ignored ->
                Map.of("status", "ok")));
        server.createContext("/api/stats", exchange -> handle(exchange, "GET", ignored ->
                backend.stats()));
        server.createContext("/api/search", exchange -> handle(exchange, "POST", input -> {
            SearchRequest request = objectMapper.readValue(input, SearchRequest.class);
            return backend.search(request);
        }));
        server.createContext("/api/search/by-trajectory", exchange -> handle(
                exchange, "POST", input -> {
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
        server.createContext("/api/deduplicate", exchange -> handle(exchange, "POST", input -> {
            DeduplicationRequest request = objectMapper.readValue(input, DeduplicationRequest.class);
            return backend.findDuplicateGroups(
                    trajectories, request.requestedThreshold(), request.requestedCandidateK());
        }));
    }

    private void handle(HttpExchange exchange, String method, ExchangeAction action)
            throws IOException {
        try {
            if (!method.equalsIgnoreCase(exchange.getRequestMethod())) {
                writeJson(exchange, 405, Map.of("error", "method not allowed"));
                return;
            }
            Object response = action.execute(exchange.getRequestBody());
            writeJson(exchange, 200, response);
        } catch (IllegalArgumentException exception) {
            writeJson(exchange, 400, Map.of("error", exception.getMessage()));
        } catch (Exception exception) {
            writeJson(exchange, 500, Map.of(
                    "error", "internal server error",
                    "detail", exception.getMessage() == null
                            ? exception.getClass().getSimpleName()
                            : exception.getMessage()));
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
