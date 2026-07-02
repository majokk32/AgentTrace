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

/**
 * Client for the localhost cuVS worker running inside WSL2.
 */
public final class CuvsTrajectorySearchBackend implements TrajectorySearchBackend {

    public static final String DEFAULT_URL = "http://127.0.0.1:8765";

    private static final TypeReference<List<SearchResult>> SEARCH_RESULTS =
            new TypeReference<>() {
            };
    private static final TypeReference<List<DuplicateGroup>> DUPLICATE_GROUPS =
            new TypeReference<>() {
            };

    private final URI workerUri;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;
    private volatile IndexStats stats =
            new IndexStats(0, 0, "cuvs-brute-force-gpu");

    public CuvsTrajectorySearchBackend(String workerUrl, ObjectMapper objectMapper) {
        if (workerUrl == null || workerUrl.isBlank()) {
            throw new IllegalArgumentException("cuVS worker URL is required");
        }
        this.workerUri = URI.create(stripTrailingSlash(workerUrl));
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    @Override
    public void rebuild(List<Trajectory> trajectories) throws IOException {
        stats = post("/rebuild", trajectories, IndexStats.class, Duration.ofMinutes(5));
    }

    @Override
    public List<SearchResult> search(SearchRequest request) throws IOException {
        return post("/search", request, SEARCH_RESULTS, Duration.ofSeconds(30));
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
        try {
            HttpResponse<byte[]> response = httpClient.send(
                    request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new IOException(
                        "cuVS worker returned HTTP " + response.statusCode()
                                + ": " + new String(
                                response.body(), java.nio.charset.StandardCharsets.UTF_8));
            }
            return response;
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while calling cuVS worker", exception);
        }
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
}
