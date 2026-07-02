package com.yikeyang.agenttrace.importer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.model.Trajectory;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class AguvisRowApiImporter {

    private static final String DATASET = "cua-lite/Aguvis";
    private static final String ROW_API =
            "https://datasets-server.huggingface.co/rows";
    private static final int PAGE_SIZE = 100;

    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;

    public AguvisRowApiImporter(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(20))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
    }

    public ImportSummary importRows(
            String config,
            String split,
            int startOffset,
            int limit,
            int embeddingDimension,
            Path output) throws IOException, InterruptedException {
        if (limit < 1 || limit > 1_000) {
            throw new IllegalArgumentException("limit must be between 1 and 1000");
        }
        if (startOffset < 0) {
            throw new IllegalArgumentException("startOffset must not be negative");
        }

        AguvisRowParser parser = new AguvisRowParser(objectMapper, embeddingDimension);
        List<Trajectory> trajectories = new ArrayList<>(limit);
        Map<String, Integer> variantCounts = new LinkedHashMap<>();
        int offset = startOffset;

        while (trajectories.size() < limit) {
            int requested = Math.min(PAGE_SIZE, limit - trajectories.size());
            JsonNode response = fetchPage(config, split, offset, requested);
            JsonNode rows = response.path("rows");
            if (!rows.isArray() || rows.isEmpty()) {
                break;
            }
            for (JsonNode row : rows) {
                Trajectory trajectory = parser.parse(row);
                trajectories.add(trajectory);
                variantCounts.merge(trajectory.app(), 1, Integer::sum);
            }
            offset += rows.size();
            if (rows.size() < requested) {
                break;
            }
        }

        if (trajectories.size() != limit) {
            throw new IOException(
                    "requested " + limit + " rows but received " + trajectories.size());
        }
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        objectMapper.writerWithDefaultPrettyPrinter()
                .writeValue(output.toFile(), trajectories);
        return new ImportSummary(
                trajectories.size(),
                embeddingDimension,
                variantCounts,
                Files.size(output));
    }

    private JsonNode fetchPage(
            String config, String split, int offset, int length)
            throws IOException, InterruptedException {
        String query = "dataset=" + encode(DATASET)
                + "&config=" + encode(config)
                + "&split=" + encode(split)
                + "&offset=" + offset
                + "&length=" + length;
        HttpRequest request = HttpRequest.newBuilder(URI.create(ROW_API + "?" + query))
                .timeout(Duration.ofSeconds(90))
                .header("Accept", "application/json")
                .header("User-Agent", "AgentTrace/0.1")
                .GET()
                .build();
        HttpResponse<InputStream> response =
                httpClient.send(request, HttpResponse.BodyHandlers.ofInputStream());
        try (InputStream body = response.body()) {
            if (response.statusCode() != 200) {
                throw new IOException(
                        "Hugging Face row API returned HTTP " + response.statusCode());
            }
            return objectMapper.readTree(body);
        }
    }

    private static String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }

    public record ImportSummary(
            int trajectoryCount,
            int embeddingDimension,
            Map<String, Integer> variantCounts,
            long outputBytes) {
    }
}

