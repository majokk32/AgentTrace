package com.yikeyang.agenttrace.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.model.DuplicateGroup;
import com.yikeyang.agenttrace.model.IndexStats;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import com.yikeyang.agenttrace.search.TrajectorySearchBackend;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import org.junit.jupiter.api.Test;

class AgentTraceServerTest {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final HttpClient httpClient = HttpClient.newHttpClient();

    @Test
    void exposesOperatorDashboardAndWorkloadTelemetry() throws Exception {
        Trajectory trajectory = new Trajectory(
                "wifi-success",
                "Turn on Wi-Fi",
                "mobile",
                "android-settings",
                true,
                List.of("tap:settings", "tap:wifi"),
                new float[]{1.0f, 0.0f});
        try (StubBackend backend = new StubBackend();
             AgentTraceServer server = new AgentTraceServer(
                     0, backend, List.of(trajectory), objectMapper)) {
            server.start();
            URI baseUri = URI.create("http://127.0.0.1:" + server.port());

            HttpResponse<String> dashboard = get(baseUri.resolve("/dashboard"));
            assertEquals(200, dashboard.statusCode());
            assertTrue(dashboard.headers().firstValue("Content-Type")
                    .orElseThrow().startsWith("text/html"));
            assertTrue(dashboard.body().contains("AgentTrace Operations"));

            HttpResponse<String> success = get(baseUri.resolve("/api/stats"));
            assertEquals(200, success.statusCode());

            assertEquals(405, get(baseUri.resolve("/api/search")).statusCode());

            JsonNode telemetry = objectMapper.readTree(get(
                    baseUri.resolve("/api/telemetry")).body());
            assertEquals("ok", telemetry.path("status").asText());
            assertEquals("test-backend", telemetry.path("backend").path("backend").asText());
            assertEquals(2, telemetry.path("workload").path("totalRequests").asLong());
            assertEquals(1, telemetry.path("workload").path("successfulRequests").asLong());
            assertEquals(1, telemetry.path("workload").path("clientErrors").asLong());
            assertEquals(0, telemetry.path("workload").path("serverErrors").asLong());
            assertEquals(1, telemetry.path("routes").path("GET /api/stats")
                    .path("requests").asLong());
            assertEquals(1, telemetry.path("routes").path("POST /api/search")
                    .path("errors").asLong());
            assertFalse(telemetry.path("routes").has("GET /api/telemetry"));
        }
    }

    private HttpResponse<String> get(URI uri) throws Exception {
        return httpClient.send(
                HttpRequest.newBuilder(uri).GET().build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private static final class StubBackend implements TrajectorySearchBackend {

        @Override
        public void rebuild(List<Trajectory> trajectories) {
        }

        @Override
        public List<SearchResult> search(SearchRequest request) {
            return List.of();
        }

        @Override
        public List<DuplicateGroup> findDuplicateGroups(
                List<Trajectory> trajectories, float threshold, int candidateK) {
            return List.of();
        }

        @Override
        public IndexStats stats() {
            return new IndexStats(1, 2, "test-backend");
        }

        @Override
        public void close() {
        }
    }
}
