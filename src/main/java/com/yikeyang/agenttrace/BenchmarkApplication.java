package com.yikeyang.agenttrace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.io.TrajectoryFileLoader;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import com.yikeyang.agenttrace.search.CuvsTrajectorySearchBackend;
import com.yikeyang.agenttrace.search.LuceneTrajectorySearchBackend;
import com.yikeyang.agenttrace.search.TrajectorySearchBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

public final class BenchmarkApplication {

    private BenchmarkApplication() {
    }

    public static void run(String[] args) throws Exception {
        Map<String, String> options = parseOptions(args);
        Path dataPath = Path.of(options.getOrDefault(
                "data", "work/aguvis-10000-minilm.json"));
        Path luceneIndexPath = Path.of(options.getOrDefault(
                "index", "data/benchmark-lucene-index"));
        Path outputPath = Path.of(options.getOrDefault(
                "output", "reports/windows-cpu-gpu-10000.json"));
        String cuvsUrl = options.getOrDefault(
                "cuvs-url", CuvsTrajectorySearchBackend.DEFAULT_URL);
        int queryCount = Integer.parseInt(options.getOrDefault("queries", "500"));
        int k = Integer.parseInt(options.getOrDefault("k", "10"));
        long seed = Long.parseLong(options.getOrDefault("seed", "42"));
        if (queryCount < 1 || queryCount > 10_000) {
            throw new IllegalArgumentException(
                    "queries must be between 1 and 10000");
        }
        if (k < 1 || k > 100) {
            throw new IllegalArgumentException("k must be between 1 and 100");
        }

        ObjectMapper objectMapper = new ObjectMapper();
        List<Trajectory> trajectories =
                new TrajectoryFileLoader(objectMapper).load(dataPath);
        if (queryCount > trajectories.size()) {
            throw new IllegalArgumentException(
                    "queries cannot exceed trajectory count");
        }
        List<SearchRequest> queries =
                selectQueries(trajectories, queryCount, k, seed);

        RunOutcome exact;
        try (CuvsTrajectorySearchBackend backend =
                     new CuvsTrajectorySearchBackend(
                             cuvsUrl, objectMapper, "brute_force")) {
            exact = runBackend(backend, trajectories, queries, null);
        }

        RunOutcome lucene;
        try (LuceneTrajectorySearchBackend backend =
                     new LuceneTrajectorySearchBackend(luceneIndexPath)) {
            lucene = runBackend(
                    backend, trajectories, queries, exact.results());
        }

        RunOutcome cagra;
        try (CuvsTrajectorySearchBackend backend =
                     new CuvsTrajectorySearchBackend(
                             cuvsUrl, objectMapper, "cagra")) {
            cagra = runBackend(
                    backend, trajectories, queries, exact.results());
        }

        BenchmarkReport report = new BenchmarkReport(
                Instant.now().toString(),
                System.getProperty("java.version"),
                System.getProperty("os.name"),
                System.getProperty("os.arch"),
                dataPath.toString(),
                trajectories.size(),
                trajectories.getFirst().embedding().length,
                queryCount,
                k,
                seed,
                exact.metrics(),
                lucene.metrics(),
                cagra.metrics(),
                "Each timed batch follows an untimed same-shape warm-up. "
                        + "Exact cuVS provides ground truth. Batch timing includes "
                        + "JSON serialization and localhost WSL2 transport; "
                        + "single-query timings measure the same end-to-end "
                        + "product path.");

        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        objectMapper.writerWithDefaultPrettyPrinter()
                .writeValue(outputPath.toFile(), report);
        System.out.printf(
                "Benchmark %d vectors, %d queries, Top-%d: "
                        + "Lucene %.4f recall at %.1f qps; "
                        + "CAGRA %.4f recall at %.1f batched qps%n",
                trajectories.size(),
                queryCount,
                k,
                lucene.metrics().meanRecallAtK(),
                lucene.metrics().batchQueriesPerSecond(),
                cagra.metrics().meanRecallAtK(),
                cagra.metrics().batchQueriesPerSecond());
        System.out.println("Wrote benchmark report to " + outputPath);
    }

    private static RunOutcome runBackend(
            TrajectorySearchBackend backend,
            List<Trajectory> trajectories,
            List<SearchRequest> queries,
            List<List<SearchResult>> groundTruth) throws Exception {
        long buildStarted = System.nanoTime();
        backend.rebuild(trajectories);
        long buildMillis = elapsedMillis(buildStarted);

        backend.search(queries.getFirst());
        backend.searchBatch(queries);
        long batchStarted = System.nanoTime();
        List<List<SearchResult>> batchResults = backend.searchBatch(queries);
        long batchNanos = System.nanoTime() - batchStarted;

        List<Long> individualLatencies = new ArrayList<>(queries.size());
        long individualStarted = System.nanoTime();
        for (SearchRequest query : queries) {
            long queryStarted = System.nanoTime();
            backend.search(query);
            individualLatencies.add(System.nanoTime() - queryStarted);
        }
        long individualNanos = System.nanoTime() - individualStarted;

        Accuracy accuracy = groundTruth == null
                ? new Accuracy(1.0, 1.0, 1.0)
                : accuracy(groundTruth, batchResults);
        double batchSeconds = batchNanos / 1_000_000_000.0;
        double individualSeconds = individualNanos / 1_000_000_000.0;
        String backendName = backend.stats().backend();
        BackendMetrics metrics = new BackendMetrics(
                backendName,
                backendName.startsWith("cuvs")
                        ? "single-http-gpu-batch"
                        : "sequential-in-process",
                buildMillis,
                batchNanos / 1_000_000.0,
                queries.size() / Math.max(batchSeconds, 0.000_001),
                batchNanos / (queries.size() * 1_000.0),
                queries.size() / Math.max(individualSeconds, 0.000_001),
                percentile(individualLatencies, 0.50) / 1_000.0,
                percentile(individualLatencies, 0.95) / 1_000.0,
                accuracy.meanRecallAtK(),
                accuracy.fullRecallRate(),
                accuracy.top1InGroundTruthRate());
        return new RunOutcome(metrics, batchResults);
    }

    private static Accuracy accuracy(
            List<List<SearchResult>> expected,
            List<List<SearchResult>> actual) {
        if (expected.size() != actual.size()) {
            throw new IllegalArgumentException(
                    "backend returned the wrong number of query result sets");
        }
        double recallTotal = 0.0;
        int fullRecall = 0;
        int top1InGroundTruth = 0;
        for (int i = 0; i < expected.size(); i++) {
            Set<String> expectedIds = expected.get(i).stream()
                    .map(SearchResult::id)
                    .collect(java.util.stream.Collectors.toCollection(HashSet::new));
            Set<String> actualIds = actual.get(i).stream()
                    .map(SearchResult::id)
                    .collect(java.util.stream.Collectors.toSet());
            long intersection = expectedIds.stream().filter(actualIds::contains).count();
            double recall = expectedIds.isEmpty()
                    ? 1.0
                    : intersection / (double) expectedIds.size();
            recallTotal += recall;
            if (intersection == expectedIds.size()) {
                fullRecall++;
            }
            if (!actual.get(i).isEmpty()
                    && expectedIds.contains(actual.get(i).getFirst().id())) {
                top1InGroundTruth++;
            }
        }
        int count = Math.max(expected.size(), 1);
        return new Accuracy(
                recallTotal / count,
                fullRecall / (double) count,
                top1InGroundTruth / (double) count);
    }

    private static List<SearchRequest> selectQueries(
            List<Trajectory> trajectories,
            int queryCount,
            int k,
            long seed) {
        List<Integer> indices = new ArrayList<>(
                IntStream.range(0, trajectories.size()).boxed().toList());
        Collections.shuffle(indices, new Random(seed));
        return indices.stream()
                .limit(queryCount)
                .map(trajectories::get)
                .map(trajectory -> new SearchRequest(
                        trajectory.embedding(), k, null, null, null))
                .toList();
    }

    private static long elapsedMillis(long started) {
        return Math.round((System.nanoTime() - started) / 1_000_000.0);
    }

    private static long percentile(List<Long> values, double quantile) {
        if (values.isEmpty()) {
            return 0L;
        }
        List<Long> sorted = values.stream().sorted().toList();
        int index = (int) Math.ceil(quantile * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
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

    private record RunOutcome(
            BackendMetrics metrics, List<List<SearchResult>> results) {
    }

    private record Accuracy(
            double meanRecallAtK,
            double fullRecallRate,
            double top1InGroundTruthRate) {
    }

    private record BenchmarkReport(
            String generatedAt,
            String javaVersion,
            String operatingSystem,
            String architecture,
            String dataPath,
            int trajectoryCount,
            int embeddingDimension,
            int queryCount,
            int k,
            long seed,
            BackendMetrics exactGroundTruth,
            BackendMetrics lucene,
            BackendMetrics cagra,
            String limitation) {
    }

    private record BackendMetrics(
            String backend,
            String batchMode,
            long indexBuildMillis,
            double batchSearchMillis,
            double batchQueriesPerSecond,
            double meanBatchQueryMicros,
            double individualQueriesPerSecond,
            double individualP50LatencyMicros,
            double individualP95LatencyMicros,
            double meanRecallAtK,
            double fullRecallRate,
            double top1InGroundTruthRate) {
    }
}
