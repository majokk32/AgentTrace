package com.yikeyang.agenttrace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.embedding.MiniLmOnnxEmbedding;
import com.yikeyang.agenttrace.io.TrajectoryFileLoader;
import com.yikeyang.agenttrace.model.DuplicateGroup;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import com.yikeyang.agenttrace.search.LuceneTrajectorySearchBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class DatasetReportApplication {

    private DatasetReportApplication() {
    }

    public static void run(String[] args) throws Exception {
        Map<String, String> options = parseOptions(args);
        Path dataPath = Path.of(options.getOrDefault(
                "data", "sample-data/aguvis-500.json"));
        Path indexPath = Path.of(options.getOrDefault(
                "index", "data/report-index"));
        Path outputPath = Path.of(options.getOrDefault(
                "output", "reports/aguvis-500-report.json"));
        float threshold = Float.parseFloat(options.getOrDefault("threshold", "0.92"));
        int candidateK = Integer.parseInt(options.getOrDefault("candidate-k", "10"));

        ObjectMapper objectMapper = new ObjectMapper();
        List<Trajectory> trajectories =
                new TrajectoryFileLoader(objectMapper).load(dataPath);

        long buildStart = System.nanoTime();
        List<DuplicateGroup> duplicateGroups;
        long buildMillis;
        long dedupMillis;
        Trajectory example = trajectories.getFirst();
        List<SearchResult> exampleResults;
        try (LuceneTrajectorySearchBackend backend =
                     new LuceneTrajectorySearchBackend(indexPath)) {
            backend.rebuild(trajectories);
            buildMillis = elapsedMillis(buildStart);

            long dedupStart = System.nanoTime();
            duplicateGroups =
                    backend.findDuplicateGroups(trajectories, threshold, candidateK);
            dedupMillis = elapsedMillis(dedupStart);

            exampleResults = backend.search(new SearchRequest(
                            example.embedding(),
                            Math.min(6, trajectories.size()),
                            example.platform(),
                            example.app(),
                            null))
                    .stream()
                    .filter(result -> !result.id().equals(example.id()))
                    .limit(5)
                    .toList();
        }

        int groupedTrajectories = duplicateGroups.stream()
                .mapToInt(group -> group.memberIds().size())
                .sum();
        int largestGroup = duplicateGroups.stream()
                .mapToInt(group -> group.memberIds().size())
                .max()
                .orElse(0);
        long imageReferences = trajectories.stream()
                .mapToLong(Trajectory::imageCount)
                .sum();
        String embeddingModel = options.getOrDefault(
                "embedding-model",
                inferEmbeddingModel(example.embedding().length));
        String limitation = options.getOrDefault(
                "limitation",
                inferLimitation(example.embedding().length));

        DatasetReport report = new DatasetReport(
                Instant.now().toString(),
                dataPath.toString(),
                trajectories.size(),
                example.embedding().length,
                embeddingModel,
                imageReferences,
                countBy(trajectories, "app"),
                countBy(trajectories, "source"),
                countBy(trajectories, "success"),
                buildMillis,
                dedupMillis,
                threshold,
                candidateK,
                duplicateGroups.size(),
                groupedTrajectories,
                largestGroup,
                new ExampleQuery(example.id(), example.instruction()),
                exampleResults,
                duplicateGroups.stream().limit(10).toList(),
                limitation);

        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(outputPath.toFile(), report);
        System.out.printf(
                "Wrote report to %s: %d trajectories, %d duplicate groups, %d ms index build%n",
                outputPath,
                trajectories.size(),
                duplicateGroups.size(),
                buildMillis);
    }

    private static Map<String, Integer> countBy(
            List<Trajectory> trajectories, String field) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Trajectory trajectory : trajectories) {
            String value = switch (field) {
                case "app" -> trajectory.app();
                case "source" -> trajectory.source();
                case "success" -> Boolean.toString(trajectory.success());
                default -> throw new IllegalArgumentException("unknown count field: " + field);
            };
            counts.merge(value, 1, Integer::sum);
        }
        return counts;
    }

    private static long elapsedMillis(long startNanos) {
        return Math.round((System.nanoTime() - startNanos) / 1_000_000.0);
    }

    private static String inferEmbeddingModel(int dimension) {
        if (dimension == MiniLmOnnxEmbedding.DIMENSION) {
            return "sentence-transformers/all-MiniLM-L6-v2-qint8-arm64";
        }
        return "AgentTrace feature-hashing baseline";
    }

    private static String inferLimitation(int dimension) {
        if (dimension == MiniLmOnnxEmbedding.DIMENSION) {
            return "Text/action-only semantic baseline: screenshot pixels are not embedded, "
                    + "and retrieval quality has not yet been measured against a labeled set.";
        }
        return "Feature hashing is a lightweight pipeline baseline, "
                + "not a semantic ML embedding.";
    }

    private static Map<String, String> parseOptions(String[] args) {
        Map<String, String> options = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String argument = args[i];
            if (!argument.startsWith("--") || i + 1 >= args.length) {
                throw new IllegalArgumentException(
                        "arguments must use --name value syntax");
            }
            options.put(argument.substring(2), args[++i]);
        }
        return options;
    }

    private record DatasetReport(
            String generatedAt,
            String dataPath,
            int trajectoryCount,
            int embeddingDimension,
            String embeddingModel,
            long imageReferences,
            Map<String, Integer> appCounts,
            Map<String, Integer> sourceCounts,
            Map<String, Integer> successCounts,
            long indexBuildMillis,
            long deduplicationMillis,
            float duplicateThreshold,
            int duplicateCandidateK,
            int duplicateGroupCount,
            int groupedTrajectoryCount,
            int largestDuplicateGroup,
            ExampleQuery exampleQuery,
            List<SearchResult> exampleResults,
            List<DuplicateGroup> duplicateExamples,
            String limitation) {
    }

    private record ExampleQuery(String id, String instruction) {
    }
}
