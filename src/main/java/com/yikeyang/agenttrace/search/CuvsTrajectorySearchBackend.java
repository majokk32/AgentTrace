package com.yikeyang.agenttrace.search;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.model.DuplicateGroup;
import com.yikeyang.agenttrace.model.IndexStats;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Locale;

/**
 * Client for the localhost cuVS worker running inside WSL2.
 */
public final class CuvsTrajectorySearchBackend implements TrajectorySearchBackend {

    public static final String DEFAULT_URL = "http://127.0.0.1:8765";

    private static final TypeReference<List<SearchResult>> SEARCH_RESULTS =
            new TypeReference<>() {
            };
    private static final TypeReference<List<List<SearchResult>>> BATCH_SEARCH_RESULTS =
            new TypeReference<>() {
            };
    private static final TypeReference<List<DuplicateGroup>> DUPLICATE_GROUPS =
            new TypeReference<>() {
            };

    private final URI workerUri;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;
    private final String algorithm;
    private volatile IndexStats stats =
            new IndexStats(0, 0, "cuvs-brute-force-gpu");

    public CuvsTrajectorySearchBackend(String workerUrl, ObjectMapper objectMapper) {
        this(workerUrl, objectMapper, "brute_force");
    }

    public CuvsTrajectorySearchBackend(
            String workerUrl, ObjectMapper objectMapper, String algorithm) {
        if (workerUrl == null || workerUrl.isBlank()) {
            throw new IllegalArgumentException("cuVS worker URL is required");
        }
        String normalizedAlgorithm = algorithm == null
                ? "brute_force"
                : algorithm.trim().toLowerCase(Locale.ROOT);
        if (!"brute_force".equals(normalizedAlgorithm)
                && !"cagra".equals(normalizedAlgorithm)) {
            throw new IllegalArgumentException(
                    "cuVS algorithm must be brute_force or cagra");
        }
        this.workerUri = URI.create(stripTrailingSlash(workerUrl));
        this.objectMapper = objectMapper;
        this.algorithm = normalizedAlgorithm;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    @Override
    public void rebuild(List<Trajectory> trajectories) throws IOException {
        stats = post(
                "/rebuild",
                new RebuildRequest(algorithm, trajectories),
                IndexStats.class,
                Duration.ofMinutes(10));
    }

    @Override
    public List<SearchResult> search(SearchRequest request) throws IOException {
        return post("/search", request, SEARCH_RESULTS, Duration.ofSeconds(30));
    }

    @Override
    public List<List<SearchResult>> searchBatch(List<SearchRequest> requests)
            throws IOException {
        if (requests == null || requests.isEmpty()) {
            return List.of();
        }
        return post(
                "/search/batch",
                new BatchSearchRequest(requests),
                BATCH_SEARCH_RESULTS,
                Duration.ofMinutes(5));
    }

    @Override
    public List<DuplicateGroup> findDuplicateGroups(
            List<Trajectory> trajectories, float threshold, int candidateK)
            throws IOException {
        return post(
                "/deduplicate",
                new DeduplicationRequest(threshold, candidateK),
                DUPLICATE_GROUPS,
                Duration.ofMinutes(5));
    }

    @Override
    public IndexStats stats() {
        return stats;
    }

    @Override
    public void close() {
        // The worker lifecycle is independent of an individual Java client.
    }

    private <T> T post(
            String path,
            Object body,
            Class<T> responseType,
            Duration timeout) throws IOException {
        HttpResponse<byte[]> response = send(path, body, timeout);
        return objectMapper.readValue(response.body(), responseType);
    }

    private <T> T post(
            String path,
            Object body,
            TypeReference<T> responseType,
            Duration timeout) throws IOException {
        HttpResponse<byte[]> response = send(path, body, timeout);
        return objectMapper.readValue(response.body(), responseType);
    }

    private HttpResponse<byte[]> send(
            String path, Object body, Duration timeout) throws IOException {
        HttpRequest request = HttpRequest.newBuilder(workerUri.resolve(path))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofByteArray(
                        objectMapper.writeValueAsBytes(body)))
                .build();
        IOException transportFailure = null;
        for (int attempt = 1; attempt <= 2; attempt++) {
            HttpResponse<byte[]> response;
            try {
                response = httpClient.send(
                        request, HttpResponse.BodyHandlers.ofByteArray());
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                throw new IOException(
                        "interrupted while calling cuVS worker", exception);
            } catch (IOException exception) {
                transportFailure = exception;
                if (attempt == 2) {
                    break;
                }
                System.err.printf(
                        "cuVS worker transport failed for %s (%s); retrying%n",
                        path, exception.getMessage());
                continue;
            }
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new IOException(
                        "cuVS worker returned HTTP " + response.statusCode()
                                + ": " + new String(
                                response.body(), java.nio.charset.StandardCharsets.UTF_8));
            }
            return response;
        }
        throw new IOException(
                "cuVS worker transport failed after retry for " + path,
                transportFailure);
    }

    private static String stripTrailingSlash(String value) {
        String result = value.trim();
        while (result.endsWith("/")) {
            result = result.substring(0, result.length() - 1);
        }
        return result;
    }

    private record DeduplicationRequest(float threshold, int candidateK) {
    }

    private record RebuildRequest(
            String algorithm, List<Trajectory> trajectories) {
    }

    private record BatchSearchRequest(List<SearchRequest> requests) {
    }
}
