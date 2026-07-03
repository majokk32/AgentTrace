package com.yikeyang.agenttrace.search;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;

public final class SearchBackendFactory {

    private SearchBackendFactory() {
    }

    public static TrajectorySearchBackend create(
            String backend,
            Path luceneIndexPath,
            String cuvsUrl,
            ObjectMapper objectMapper) throws IOException {
        return create(
                backend, luceneIndexPath, cuvsUrl, "brute_force", objectMapper);
    }

    public static TrajectorySearchBackend create(
            String backend,
            Path luceneIndexPath,
            String cuvsUrl,
            String cuvsAlgorithm,
            ObjectMapper objectMapper) throws IOException {
        String name = backend == null
                ? "lucene"
                : backend.trim().toLowerCase(Locale.ROOT);
        return switch (name) {
            case "lucene", "lucene-hnsw" ->
                    new LuceneTrajectorySearchBackend(luceneIndexPath);
            case "cuvs", "cuvs-brute-force-gpu" ->
                    new CuvsTrajectorySearchBackend(
                            cuvsUrl, objectMapper, cuvsAlgorithm);
            case "cuvs-cagra", "cuvs-cagra-gpu" ->
                    new CuvsTrajectorySearchBackend(
                            cuvsUrl, objectMapper, "cagra");
            default -> throw new IllegalArgumentException(
                    "unknown backend " + backend
                            + "; expected lucene, cuvs, or cuvs-cagra");
        };
    }
}
