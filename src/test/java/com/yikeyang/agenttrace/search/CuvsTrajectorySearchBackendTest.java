package com.yikeyang.agenttrace.search;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import com.yikeyang.agenttrace.model.DuplicateGroup;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.List;
import org.junit.jupiter.api.Test;

class CuvsTrajectorySearchBackendTest {

    @Test
    void proxiesBackendOperationsToWorker() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/rebuild", exchange -> writeJson(
                mapper, exchange, """
                        {"trajectoryCount":1,"vectorDimension":2,
                         "backend":"cuvs-brute-force-gpu"}
                        """));
        server.createContext("/search", exchange -> writeJson(
                mapper, exchange, """
                        [{"id":"one","instruction":"test","platform":"mobile",
                          "app":"settings","success":true,"actions":["tap"],
                          "source":"fixture","sourceId":"one","imageCount":0,
                          "score":1.0}]
                        """));
        server.createContext("/deduplicate", exchange -> writeJson(
                mapper, exchange, """
                        [{"canonicalId":"one","memberIds":["one","two"],
                          "meanCosineSimilarity":0.99}]
                        """));
        server.start();

        try (CuvsTrajectorySearchBackend backend =
                     new CuvsTrajectorySearchBackend(
                             "http://127.0.0.1:" + server.getAddress().getPort(),
                             mapper)) {
            List<Trajectory> trajectories = List.of(new Trajectory(
                    "one",
                    "test",
                    "mobile",
                    "settings",
                    true,
                    List.of("tap"),
                    new float[] {1.0f, 0.0f},
                    "fixture",
                    "one",
                    0));
            backend.rebuild(trajectories);
            assertEquals("cuvs-brute-force-gpu", backend.stats().backend());

            List<SearchResult> results = backend.search(new SearchRequest(
                    new float[] {1.0f, 0.0f}, 1, "mobile", "settings", true));
            assertEquals("one", results.getFirst().id());

            List<DuplicateGroup> groups =
                    backend.findDuplicateGroups(trajectories, 0.9f, 2);
            assertEquals(List.of("one", "two"), groups.getFirst().memberIds());
        } finally {
            server.stop(0);
        }
    }

    private static void writeJson(
            ObjectMapper mapper, HttpExchange exchange, String json)
            throws IOException {
        exchange.getRequestBody().readAllBytes();
        byte[] payload = mapper.readTree(json).toString().getBytes(
                java.nio.charset.StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, payload.length);
        exchange.getResponseBody().write(payload);
        exchange.close();
    }
}
