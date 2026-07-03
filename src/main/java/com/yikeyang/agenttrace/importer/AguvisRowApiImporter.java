package com.yikeyang.agenttrace.importer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.model.Trajectory;
import java.io.IOException;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public final class AguvisRowApiImporter {

    private static final String DATASET = "cua-lite/Aguvis";
    private static final String ROW_API =
            "https://datasets-server.huggingface.co/rows";
    private static final int PAGE_SIZE = 100;
    private static final int RESPONSE_TIMEOUT_SECONDS = 8;

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
        if (limit < 1 || limit > 100_000) {
            throw new IllegalArgumentException("limit must be between 1 and 100000");
        }
        if (startOffset < 0) {
            throw new IllegalArgumentException("startOffset must not be negative");
        }

        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path partialOutput = output.resolveSibling(
                output.getFileName() + ".partial");
        AguvisRowParser parser = new AguvisRowParser(objectMapper, embeddingDimension);
        List<Trajectory> trajectories = new ArrayList<>(limit);
        Map<String, Integer> variantCounts = new LinkedHashMap<>();
        int offset = startOffset;
        int nextProgress = 1_000;
        if (Files.exists(partialOutput)) {
            List<Trajectory> checkpoint = objectMapper
                    .readerForListOf(Trajectory.class)
                    .readValue(partialOutput.toFile());
            if (checkpoint.size() > limit) {
                throw new IOException(
                        "checkpoint contains more rows than the requested limit");
            }
            trajectories.addAll(checkpoint);
            checkpoint.forEach(trajectory ->
                    variantCounts.merge(trajectory.app(), 1, Integer::sum));
            offset += checkpoint.size();
            nextProgress = (checkpoint.size() / 1_000 + 1) * 1_000;
            System.out.printf(
                    "Resuming AGUVIS import from %,d checkpointed rows%n",
                    checkpoint.size());
        }

        while (trajectories.size() < limit) {
            int requested = Math.min(PAGE_SIZE, limit - trajectories.size());
            JsonNode rows = fetchPage(config, split, offset, requested)
                    .path("rows");
            if (!rows.isArray() || rows.isEmpty()) {
                break;
            }
            for (JsonNode row : rows) {
                Trajectory trajectory = parser.parse(row);
                trajectories.add(trajectory);
                variantCounts.merge(trajectory.app(), 1, Integer::sum);
            }
            offset += rows.size();
            if (trajectories.size() >= nextProgress
                    || trajectories.size() == limit) {
                System.out.printf(
                        "Fetched %,d of %,d AGUVIS rows%n",
                        trajectories.size(), limit);
                System.out.flush();
                objectMapper.writerWithDefaultPrettyPrinter()
                        .writeValue(partialOutput.toFile(), trajectories);
                while (nextProgress <= trajectories.size()) {
                    nextProgress += 1_000;
                }
            }
        }

        if (trajectories.size() != limit) {
            throw new IOException(
                    "requested " + limit + " rows but received " + trajectories.size());
        }
        objectMapper.writerWithDefaultPrettyPrinter()
                .writeValue(output.toFile(), trajectories);
        Files.deleteIfExists(partialOutput);
        return new ImportSummary(
                trajectories.size(),
                embeddingDimension,
                variantCounts,
                Files.size(output));
    }

    public ImportSummary importRowsFile(
            Path rowsFile,
            int limit,
            int embeddingDimension,
            Path output) throws IOException {
        JsonNode rows = objectMapper.readTree(rowsFile.toFile());
        if (!rows.isArray()) {
            throw new IOException("AGUVIS rows file must contain a JSON array");
        }
        if (limit < 1 || limit > rows.size()) {
            throw new IOException(
                    "requested " + limit + " rows but file contains " + rows.size());
        }
        AguvisRowParser parser = new AguvisRowParser(objectMapper, embeddingDimension);
        List<Trajectory> trajectories = new ArrayList<>(limit);
        Map<String, Integer> variantCounts = new LinkedHashMap<>();
        for (int index = 0; index < limit; index++) {
            Trajectory trajectory = parser.parse(rows.get(index));
            trajectories.add(trajectory);
            variantCounts.merge(trajectory.app(), 1, Integer::sum);
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
        IOException failure;
        while (true) {
            try {
                return fetchPageOnce(config, split, offset, length);
            } catch (IOException caught) {
                if (!String.valueOf(caught.getMessage()).contains("HTTP 429")) {
                    failure = caught;
                    break;
                }
                System.err.printf(
                        "AGUVIS page at offset %,d was rate limited; "
                                + "retrying after 15 seconds%n",
                        offset);
                System.err.flush();
                Thread.sleep(15_000);
            }
        }
        if (length == 1) {
            throw failure;
        }
        int firstLength = length / 2;
        int secondLength = length - firstLength;
        System.err.printf(
                "AGUVIS page at offset %,d with %d rows failed (%s); "
                        + "splitting into %d and %d rows%n",
                offset,
                length,
                failure.getMessage(),
                firstLength,
                secondLength);
        System.err.flush();
        JsonNode first = fetchPage(
                config, split, offset, firstLength);
        JsonNode second = fetchPage(
                config, split, offset + firstLength, secondLength);
        var merged = objectMapper.createObjectNode();
        var rows = merged.putArray("rows");
        first.path("rows").forEach(rows::add);
        second.path("rows").forEach(rows::add);
        return merged;
    }

    private JsonNode fetchPageOnce(
            String config, String split, int offset, int length)
            throws IOException, InterruptedException {
        String query = "dataset=" + encode(DATASET)
                + "&config=" + encode(config)
                + "&split=" + encode(split)
                + "&offset=" + offset
                + "&length=" + length;
        HttpRequest request = HttpRequest.newBuilder(URI.create(ROW_API + "?" + query))
                .timeout(Duration.ofSeconds(180))
                .header("Accept", "application/json")
                .header("User-Agent", "AgentTrace/0.1")
                .GET()
                .build();
        Future<HttpResponse<byte[]>> future = httpClient.sendAsync(
                request, HttpResponse.BodyHandlers.ofByteArray());
        HttpResponse<byte[]> response;
        try {
            response = future.get(
                    RESPONSE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (TimeoutException failure) {
            future.cancel(true);
            throw new IOException(
                    "response exceeded " + RESPONSE_TIMEOUT_SECONDS + " seconds",
                    failure);
        } catch (ExecutionException failure) {
            Throwable cause = failure.getCause();
            if (cause instanceof IOException ioFailure) {
                throw ioFailure;
            }
            throw new IOException("AGUVIS page fetch failed", cause);
        }
        if (response.statusCode() != 200) {
            throw new IOException(
                    "Hugging Face row API returned HTTP " + response.statusCode());
        }
        return objectMapper.readTree(response.body());
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
