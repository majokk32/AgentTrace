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
        String name = backend == null
                ? "lucene"
                : backend.trim().toLowerCase(Locale.ROOT);
        return switch (name) {
            case "lucene", "lucene-hnsw" ->
                    new LuceneTrajectorySearchBackend(luceneIndexPath);
            case "cuvs", "cuvs-brute-force-gpu" ->
                    new CuvsTrajectorySearchBackend(cuvsUrl, objectMapper);
            default -> throw new IllegalArgumentException(
                    "unknown backend " + backend + "; expected lucene or cuvs");
        };
    }
}
