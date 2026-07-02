package com.yikeyang.agenttrace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.evaluation.FailurePairSet;
import com.yikeyang.agenttrace.evaluation.FailurePairSet.FailurePair;
import com.yikeyang.agenttrace.io.TrajectoryFileLoader;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import com.yikeyang.agenttrace.search.CuvsTrajectorySearchBackend;
import com.yikeyang.agenttrace.search.SearchBackendFactory;
import com.yikeyang.agenttrace.search.TrajectorySearchBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class RecoveryEvaluationApplication {

    private RecoveryEvaluationApplication() {
    }

    public static void run(String[] args) throws Exception {
        Map<String, String> options = parseOptions(args);
        Path dataPath = Path.of(options.getOrDefault(
                "data", "sample-data/aguvis-500-plus-50-failures.json"));
        Path pairsPath = Path.of(options.getOrDefault(
                "pairs", "labels/aguvis-failure-pairs-50.json"));
        Path indexPath = Path.of(options.getOrDefault(
                "index", "data/recovery-evaluation-index"));
        Path outputPath = Path.of(options.getOrDefault(
                "output", "reports/recovery-evaluation.json"));
        int k = Integer.parseInt(options.getOrDefault("k", "5"));
        String backendName = options.getOrDefault("backend", "lucene");
        String cuvsUrl = options.getOrDefault(
                "cuvs-url", CuvsTrajectorySearchBackend.DEFAULT_URL);
        if (k < 1 || k > 100) {
            throw new IllegalArgumentException("k must be between 1 and 100");
        }

        ObjectMapper objectMapper = new ObjectMapper();
        List<Trajectory> trajectories =
                new TrajectoryFileLoader(objectMapper).load(dataPath);
        FailurePairSet pairSet =
                objectMapper.readValue(pairsPath.toFile(), FailurePairSet.class);
        Map<String, Trajectory> byId = indexById(trajectories);
        validatePairs(pairSet, byId);

        List<RecoveryQuery> queries = new ArrayList<>();
        List<Long> latenciesMicros = new ArrayList<>();
        long buildStarted = System.nanoTime();
        long indexBuildMillis;
        String backend;
        try (TrajectorySearchBackend searchBackend = SearchBackendFactory.create(
                backendName, indexPath, cuvsUrl, objectMapper)) {
            searchBackend.rebuild(trajectories);
            indexBuildMillis = elapsedMillis(buildStarted);
            backend = searchBackend.stats().backend();
            for (FailurePair pair : pairSet.pairs()) {
                Trajectory failure = byId.get(pair.failureId());
                long searchStarted = System.nanoTime();
                List<SearchResult> rawResults = searchBackend.search(new SearchRequest(
                        failure.embedding(),
                        Math.min(100, k + 1),
                        failure.platform(),
                        failure.app(),
                        true));
                long latencyMicros = Math.round(
                        (System.nanoTime() - searchStarted) / 1_000.0);
                latenciesMicros.add(latencyMicros);
                int parentRank = 0;
                for (int rank = 0; rank < rawResults.size(); rank++) {
                    if (rawResults.get(rank).id().equals(pair.successfulId())) {
                        parentRank = rank + 1;
                        break;
                    }
                }
                List<SearchResult> parentExcludedResults = rawResults.stream()
                        .filter(result -> !result.id().equals(pair.successfulId()))
                        .limit(k)
                        .toList();
                Set<String> alternativeIds =
                        new HashSet<>(pair.alternativeSuccessfulIds());
                int alternativesRetrieved = 0;
                int firstAlternativeRank = 0;
                for (int rank = 0; rank < parentExcludedResults.size(); rank++) {
                    if (alternativeIds.contains(
                            parentExcludedResults.get(rank).id())) {
                        alternativesRetrieved++;
                        if (firstAlternativeRank == 0) {
                            firstAlternativeRank = rank + 1;
                        }
                    }
                }
                double alternativeRecall = alternativeIds.isEmpty()
                        ? 0.0
                        : alternativesRetrieved / (double) alternativeIds.size();
                queries.add(new RecoveryQuery(
                        pair.failureId(),
                        failure.instruction(),
                        pair.successfulId(),
                        parentRank,
                        alternativeIds.size(),
                        alternativesRetrieved,
                        alternativeRecall,
                        firstAlternativeRank,
                        firstAlternativeRank == 0
                                ? 0.0
                                : 1.0 / firstAlternativeRank,
                        latencyMicros,
                        pair.originalActionCount(),
                        pair.remainingActionCount(),
                        parentExcludedResults.stream().map(result -> new RankedResult(
                                result.id(), result.instruction(), result.score()))
                                .toList()));
            }
        }

        double parentRecallAt1 = queries.stream()
                .filter(query -> query.parentRank() == 1)
                .count() / (double) queries.size();
        double parentRecallAtK = queries.stream()
                .filter(query -> query.parentRank() > 0
                        && query.parentRank() <= k)
                .count() / (double) queries.size();
        List<RecoveryQuery> alternativeQueries = queries.stream()
                .filter(query -> query.alternativeRelevantCount() > 0)
                .toList();
        double alternativeRecallAtK = alternativeQueries.stream()
                .mapToDouble(RecoveryQuery::alternativeRecallAtK)
                .average()
                .orElse(0.0);
        double alternativeHitRateAtK = alternativeQueries.stream()
                .filter(query -> query.firstAlternativeRank() > 0)
                .count() / (double) Math.max(alternativeQueries.size(), 1);
        double alternativeMeanReciprocalRank = alternativeQueries.stream()
                .mapToDouble(RecoveryQuery::reciprocalRank)
                .average()
                .orElse(0.0);
        RecoveryReport report = new RecoveryReport(
                Instant.now().toString(),
                System.getProperty("java.version"),
                System.getProperty("os.name"),
                System.getProperty("os.arch"),
                dataPath.toString(),
                pairsPath.toString(),
                pairSet.name(),
                pairSet.version(),
                pairSet.generationPolicy(),
                backend,
                trajectories.size(),
                pairSet.pairs().size(),
                k,
                indexBuildMillis,
                parentRecallAt1,
                parentRecallAtK,
                alternativeQueries.size(),
                alternativeRecallAtK,
                alternativeHitRateAtK,
                alternativeMeanReciprocalRank,
                percentile(latenciesMicros, 0.50),
                percentile(latenciesMicros, 0.95),
                alternativeQueries.stream()
                        .sorted(Comparator.comparingInt((RecoveryQuery query) ->
                                query.firstAlternativeRank() == 0
                                        ? Integer.MAX_VALUE
                                        : query.firstAlternativeRank()).reversed())
                        .limit(15)
                        .toList(),
                "Controlled truncation failures test retrieval behavior but are "
                        + "not a substitute for naturally collected failed sessions. "
                        + "The alternative-intent metrics exclude each failure's "
                        + "exact parent trajectory from ranked results.");
        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        objectMapper.writerWithDefaultPrettyPrinter()
                .writeValue(outputPath.toFile(), report);
        System.out.printf(
                "Recovery evaluation: parent Recall@1 %.4f; parent-excluded "
                        + "intent Recall@%d %.4f, HitRate@%d %.4f, MRR %.4f%n",
                parentRecallAt1,
                k,
                alternativeRecallAtK,
                k,
                alternativeHitRateAtK,
                alternativeMeanReciprocalRank);
        System.out.println("Wrote recovery report to " + outputPath);
    }

    private static Map<String, Trajectory> indexById(List<Trajectory> trajectories) {
        Map<String, Trajectory> byId = new LinkedHashMap<>();
        for (Trajectory trajectory : trajectories) {
            if (byId.put(trajectory.id(), trajectory) != null) {
                throw new IllegalArgumentException(
                        "duplicate trajectory id " + trajectory.id());
            }
        }
        return byId;
    }

    private static void validatePairs(
            FailurePairSet pairSet, Map<String, Trajectory> byId) {
        if (pairSet.pairs().isEmpty()) {
            throw new IllegalArgumentException("failure pair set must not be empty");
        }
        for (FailurePair pair : pairSet.pairs()) {
            Trajectory failure = byId.get(pair.failureId());
            Trajectory success = byId.get(pair.successfulId());
            if (failure == null || success == null) {
                throw new IllegalArgumentException(
                        "pair references a missing trajectory: " + pair);
            }
            if (failure.success()) {
                throw new IllegalArgumentException(
                        "failure trajectory is marked successful: " + failure.id());
            }
            if (!success.success()) {
                throw new IllegalArgumentException(
                        "successful trajectory is marked failed: " + success.id());
            }
            if (!failure.sourceId().equals(success.id())) {
                throw new IllegalArgumentException(
                        "failure provenance does not point to successful trajectory");
            }
        }
    }

    private static long percentile(List<Long> values, double quantile) {
        List<Long> sorted = values.stream().sorted().toList();
        int index = (int) Math.ceil(quantile * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private static long elapsedMillis(long started) {
        return Math.round((System.nanoTime() - started) / 1_000_000.0);
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

    private record RecoveryReport(
            String generatedAt,
            String javaVersion,
            String operatingSystem,
            String architecture,
            String dataPath,
            String pairsPath,
            String pairSet,
            String pairVersion,
            String generationPolicy,
            String backend,
            int trajectoryCount,
            int failureQueryCount,
            int k,
            long indexBuildMillis,
            double parentRecallAt1,
            double parentRecallAtK,
            int alternativeIntentQueryCount,
            double parentExcludedIntentRecallAtK,
            double parentExcludedIntentHitRateAtK,
            double parentExcludedIntentMeanReciprocalRank,
            long p50LatencyMicros,
            long p95LatencyMicros,
            List<RecoveryQuery> hardestQueries,
            String limitation) {
    }

    private record RecoveryQuery(
            String failureId,
            String instruction,
            String expectedSuccessfulId,
            int parentRank,
            int alternativeRelevantCount,
            int alternativesRetrieved,
            double alternativeRecallAtK,
            int firstAlternativeRank,
            double reciprocalRank,
            long latencyMicros,
            int originalActionCount,
            int remainingActionCount,
            List<RankedResult> results) {
    }

    private record RankedResult(String id, String instruction, float score) {
    }
}
