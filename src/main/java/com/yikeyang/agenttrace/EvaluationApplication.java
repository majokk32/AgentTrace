package com.yikeyang.agenttrace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.evaluation.EvaluationLabels;
import com.yikeyang.agenttrace.evaluation.EvaluationLabels.LabelGroup;
import com.yikeyang.agenttrace.evaluation.EvaluationLabels.NegativePair;
import com.yikeyang.agenttrace.io.TrajectoryFileLoader;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import com.yikeyang.agenttrace.search.CuvsTrajectorySearchBackend;
import com.yikeyang.agenttrace.search.SearchBackendFactory;
import com.yikeyang.agenttrace.search.TrajectorySearchBackend;
import com.yikeyang.agenttrace.search.VectorMath;
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

public final class EvaluationApplication {

    private EvaluationApplication() {
    }

    public static void run(String[] args) throws Exception {
        Map<String, String> options = parseOptions(args);
        Path dataPath = Path.of(required(options, "data"));
        Path labelsPath = Path.of(options.getOrDefault(
                "labels", "labels/aguvis-500-intents-v1.json"));
        Path indexPath = Path.of(options.getOrDefault(
                "index", "data/evaluation-index"));
        Path outputPath = Path.of(options.getOrDefault(
                "output", "reports/evaluation.json"));
        String embeddingModel = options.getOrDefault("embedding-model", "unspecified");
        int k = Integer.parseInt(options.getOrDefault("k", "5"));
        float duplicateThreshold = Float.parseFloat(
                options.getOrDefault("duplicate-threshold", "0.92"));
        String backendName = options.getOrDefault("backend", "lucene");
        String cuvsUrl = options.getOrDefault(
                "cuvs-url", CuvsTrajectorySearchBackend.DEFAULT_URL);
        String cuvsAlgorithm = options.getOrDefault(
                "cuvs-algorithm", "brute_force");
        if (k < 1 || k > 100) {
            throw new IllegalArgumentException("k must be between 1 and 100");
        }
        if (duplicateThreshold < -1.0f || duplicateThreshold > 1.0f) {
            throw new IllegalArgumentException(
                    "duplicate-threshold must be between -1 and 1");
        }

        ObjectMapper objectMapper = new ObjectMapper();
        List<Trajectory> trajectories =
                new TrajectoryFileLoader(objectMapper).load(dataPath);
        EvaluationLabels labels =
                objectMapper.readValue(labelsPath.toFile(), EvaluationLabels.class);
        Map<String, Trajectory> byId = indexById(trajectories);
        validateLabels(labels, byId);

        long buildStarted = System.nanoTime();
        RetrievalMetrics retrievalMetrics;
        long indexBuildMillis;
        String backend;
        try (TrajectorySearchBackend searchBackend = SearchBackendFactory.create(
                backendName, indexPath, cuvsUrl, cuvsAlgorithm, objectMapper)) {
            searchBackend.rebuild(trajectories);
            indexBuildMillis = elapsedMillis(buildStarted);
            backend = searchBackend.stats().backend();
            retrievalMetrics = evaluateRetrieval(
                    searchBackend, labels, byId, k);
        }
        DedupMetrics dedupMetrics =
                evaluateDeduplication(labels, byId, duplicateThreshold);

        EvaluationReport report = new EvaluationReport(
                Instant.now().toString(),
                System.getProperty("java.version"),
                System.getProperty("os.name"),
                System.getProperty("os.arch"),
                dataPath.toString(),
                labelsPath.toString(),
                labels.name(),
                labels.version(),
                labels.annotationPolicy(),
                embeddingModel,
                backend,
                trajectories.size(),
                trajectories.getFirst().embedding().length,
                indexBuildMillis,
                retrievalMetrics,
                dedupMetrics);
        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        objectMapper.writerWithDefaultPrettyPrinter()
                .writeValue(outputPath.toFile(), report);
        System.out.printf(
                "Evaluation %s: Recall@%d %.4f, HitRate@%d %.4f, MRR %.4f, "
                        + "dedup F1 %.4f%n",
                embeddingModel,
                k,
                retrievalMetrics.recallAtK(),
                k,
                retrievalMetrics.hitRateAtK(),
                retrievalMetrics.meanReciprocalRank(),
                dedupMetrics.f1());
        System.out.println("Wrote evaluation report to " + outputPath);
    }

    private static RetrievalMetrics evaluateRetrieval(
            TrajectorySearchBackend backend,
            EvaluationLabels labels,
            Map<String, Trajectory> byId,
            int k) throws Exception {
        List<QueryResult> queryResults = new ArrayList<>();
        List<Long> latenciesMicros = new ArrayList<>();
        for (LabelGroup group : labels.retrievalGroups()) {
            Set<String> groupIds = Set.copyOf(group.memberIds());
            for (String queryId : group.memberIds()) {
                Trajectory query = byId.get(queryId);
                Set<String> relevantIds = new HashSet<>(groupIds);
                relevantIds.remove(queryId);
                long started = System.nanoTime();
                List<SearchResult> results = backend.search(new SearchRequest(
                                query.embedding(),
                                Math.min(100, k + 1),
                                query.platform(),
                                query.app(),
                                null))
                        .stream()
                        .filter(result -> !result.id().equals(queryId))
                        .limit(k)
                        .toList();
                long latencyMicros = Math.round(
                        (System.nanoTime() - started) / 1_000.0);
                latenciesMicros.add(latencyMicros);

                int relevantRetrieved = 0;
                int firstRelevantRank = 0;
                for (int rank = 0; rank < results.size(); rank++) {
                    if (relevantIds.contains(results.get(rank).id())) {
                        relevantRetrieved++;
                        if (firstRelevantRank == 0) {
                            firstRelevantRank = rank + 1;
                        }
                    }
                }
                double recall = relevantIds.isEmpty()
                        ? 0.0
                        : relevantRetrieved / (double) relevantIds.size();
                double precision = relevantRetrieved / (double) k;
                double reciprocalRank =
                        firstRelevantRank == 0 ? 0.0 : 1.0 / firstRelevantRank;
                queryResults.add(new QueryResult(
                        group.label(),
                        queryId,
                        query.instruction(),
                        relevantIds.size(),
                        relevantRetrieved,
                        recall,
                        precision,
                        firstRelevantRank,
                        reciprocalRank,
                        latencyMicros,
                        results.stream().map(result -> new RankedResult(
                                result.id(), result.instruction(), result.score()))
                                .toList()));
            }
        }
        return new RetrievalMetrics(
                k,
                queryResults.size(),
                labels.retrievalGroups().size(),
                average(queryResults.stream().map(QueryResult::recallAtK).toList()),
                average(queryResults.stream().map(QueryResult::precisionAtK).toList()),
                average(queryResults.stream()
                        .map(result -> result.firstRelevantRank() == 0 ? 0.0 : 1.0)
                        .toList()),
                average(queryResults.stream()
                        .map(QueryResult::reciprocalRank)
                        .toList()),
                percentile(latenciesMicros, 0.50),
                percentile(latenciesMicros, 0.95),
                queryResults.stream()
                        .sorted(Comparator.comparingDouble(QueryResult::recallAtK))
                        .limit(10)
                        .toList());
    }

    private static DedupMetrics evaluateDeduplication(
            EvaluationLabels labels,
            Map<String, Trajectory> byId,
            float threshold) {
        List<LabeledPair> pairs = new ArrayList<>();
        for (LabelGroup group : labels.duplicateGroups()) {
            for (int left = 0; left < group.memberIds().size(); left++) {
                for (int right = left + 1; right < group.memberIds().size(); right++) {
                    pairs.add(new LabeledPair(
                            group.memberIds().get(left),
                            group.memberIds().get(right),
                            true,
                            group.label()));
                }
            }
        }
        List<GroupedMember> duplicateMembers = labels.duplicateGroups().stream()
                .flatMap(group -> group.memberIds().stream()
                        .map(id -> new GroupedMember(id, group.label())))
                .toList();
        for (int left = 0; left < duplicateMembers.size(); left++) {
            for (int right = left + 1; right < duplicateMembers.size(); right++) {
                GroupedMember leftMember = duplicateMembers.get(left);
                GroupedMember rightMember = duplicateMembers.get(right);
                if (!leftMember.group().equals(rightMember.group())) {
                    pairs.add(new LabeledPair(
                            leftMember.id(),
                            rightMember.id(),
                            false,
                            "cross-group:" + leftMember.group()
                                    + "!=" + rightMember.group()));
                }
            }
        }
        for (NegativePair pair : labels.negativePairs()) {
            pairs.add(new LabeledPair(
                    pair.leftId(), pair.rightId(), false, pair.rationale()));
        }

        int truePositive = 0;
        int falsePositive = 0;
        int trueNegative = 0;
        int falseNegative = 0;
        List<PairError> errors = new ArrayList<>();
        for (LabeledPair pair : pairs) {
            Trajectory left = byId.get(pair.leftId());
            Trajectory right = byId.get(pair.rightId());
            float similarity = VectorMath.cosineSimilarity(
                    left.embedding(), right.embedding());
            boolean predictedDuplicate = similarity >= threshold;
            if (pair.duplicate() && predictedDuplicate) {
                truePositive++;
            } else if (pair.duplicate()) {
                falseNegative++;
                errors.add(pairError(pair, left, right, similarity));
            } else if (predictedDuplicate) {
                falsePositive++;
                errors.add(pairError(pair, left, right, similarity));
            } else {
                trueNegative++;
            }
        }
        double precision = safeDivide(truePositive, truePositive + falsePositive);
        double recall = safeDivide(truePositive, truePositive + falseNegative);
        double f1 = precision + recall == 0.0
                ? 0.0
                : 2.0 * precision * recall / (precision + recall);
        List<ThresholdScore> thresholdSweep = new ArrayList<>();
        for (int value = 60; value <= 99; value++) {
            thresholdSweep.add(scoreThreshold(pairs, byId, value / 100.0f));
        }
        ThresholdScore calibratedBest = thresholdSweep.stream()
                .max(Comparator.comparingDouble(ThresholdScore::f1)
                        .thenComparingDouble(ThresholdScore::precision)
                        .thenComparingDouble(ThresholdScore::threshold))
                .orElseThrow();
        return new DedupMetrics(
                threshold,
                pairs.size(),
                truePositive,
                falsePositive,
                trueNegative,
                falseNegative,
                precision,
                recall,
                f1,
                calibratedBest,
                thresholdSweep,
                errors.stream()
                        .sorted(Comparator.comparingDouble(PairError::similarity).reversed())
                        .limit(20)
                        .toList());
    }

    private static ThresholdScore scoreThreshold(
            List<LabeledPair> pairs,
            Map<String, Trajectory> byId,
            float threshold) {
        int truePositive = 0;
        int falsePositive = 0;
        int trueNegative = 0;
        int falseNegative = 0;
        for (LabeledPair pair : pairs) {
            Trajectory left = byId.get(pair.leftId());
            Trajectory right = byId.get(pair.rightId());
            boolean predictedDuplicate = VectorMath.cosineSimilarity(
                    left.embedding(), right.embedding()) >= threshold;
            if (pair.duplicate() && predictedDuplicate) {
                truePositive++;
            } else if (pair.duplicate()) {
                falseNegative++;
            } else if (predictedDuplicate) {
                falsePositive++;
            } else {
                trueNegative++;
            }
        }
        double precision = safeDivide(truePositive, truePositive + falsePositive);
        double recall = safeDivide(truePositive, truePositive + falseNegative);
        double f1 = precision + recall == 0.0
                ? 0.0
                : 2.0 * precision * recall / (precision + recall);
        return new ThresholdScore(
                threshold,
                truePositive,
                falsePositive,
                trueNegative,
                falseNegative,
                precision,
                recall,
                f1);
    }

    private static PairError pairError(
            LabeledPair pair,
            Trajectory left,
            Trajectory right,
            float similarity) {
        return new PairError(
                pair.duplicate() ? "false-negative" : "false-positive",
                pair.label(),
                left.id(),
                left.instruction(),
                right.id(),
                right.instruction(),
                similarity);
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

    private static void validateLabels(
            EvaluationLabels labels, Map<String, Trajectory> byId) {
        if (labels.retrievalGroups().isEmpty()) {
            throw new IllegalArgumentException("retrievalGroups must not be empty");
        }
        for (LabelGroup group : labels.retrievalGroups()) {
            validateGroup(group, byId, "retrieval");
        }
        for (LabelGroup group : labels.duplicateGroups()) {
            validateGroup(group, byId, "duplicate");
        }
        List<String> duplicateMemberIds = labels.duplicateGroups().stream()
                .flatMap(group -> group.memberIds().stream())
                .toList();
        if (new HashSet<>(duplicateMemberIds).size() != duplicateMemberIds.size()) {
            throw new IllegalArgumentException(
                    "a trajectory cannot belong to multiple duplicate groups");
        }
        for (NegativePair pair : labels.negativePairs()) {
            requireTrajectory(pair.leftId(), byId);
            requireTrajectory(pair.rightId(), byId);
            if (pair.leftId().equals(pair.rightId())) {
                throw new IllegalArgumentException(
                        "negative pair cannot contain the same id twice");
            }
        }
    }

    private static void validateGroup(
            LabelGroup group,
            Map<String, Trajectory> byId,
            String kind) {
        if (group.memberIds().size() < 2) {
            throw new IllegalArgumentException(
                    kind + " group " + group.label() + " needs at least two members");
        }
        if (new HashSet<>(group.memberIds()).size() != group.memberIds().size()) {
            throw new IllegalArgumentException(
                    kind + " group " + group.label() + " contains duplicate ids");
        }
        for (String id : group.memberIds()) {
            requireTrajectory(id, byId);
        }
    }

    private static void requireTrajectory(
            String id, Map<String, Trajectory> byId) {
        if (!byId.containsKey(id)) {
            throw new IllegalArgumentException(
                    "label references missing trajectory " + id);
        }
    }

    private static double average(List<Double> values) {
        return values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    private static long percentile(List<Long> values, double quantile) {
        if (values.isEmpty()) {
            return 0L;
        }
        List<Long> sorted = values.stream().sorted().toList();
        int index = (int) Math.ceil(quantile * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private static double safeDivide(int numerator, int denominator) {
        return denominator == 0 ? 0.0 : numerator / (double) denominator;
    }

    private static long elapsedMillis(long started) {
        return Math.round((System.nanoTime() - started) / 1_000_000.0);
    }

    private static String required(Map<String, String> options, String name) {
        String value = options.get(name);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("--" + name + " is required");
        }
        return value;
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

    private record LabeledPair(
            String leftId, String rightId, boolean duplicate, String label) {
    }

    private record GroupedMember(String id, String group) {
    }

    private record EvaluationReport(
            String generatedAt,
            String javaVersion,
            String operatingSystem,
            String architecture,
            String dataPath,
            String labelsPath,
            String labelSet,
            String labelVersion,
            String annotationPolicy,
            String embeddingModel,
            String backend,
            int trajectoryCount,
            int embeddingDimension,
            long indexBuildMillis,
            RetrievalMetrics retrieval,
            DedupMetrics deduplication) {
    }

    private record RetrievalMetrics(
            int k,
            int queryCount,
            int intentGroupCount,
            double recallAtK,
            double precisionAtK,
            double hitRateAtK,
            double meanReciprocalRank,
            long p50LatencyMicros,
            long p95LatencyMicros,
            List<QueryResult> lowestRecallQueries) {
    }

    private record QueryResult(
            String label,
            String queryId,
            String instruction,
            int relevantCount,
            int relevantRetrieved,
            double recallAtK,
            double precisionAtK,
            int firstRelevantRank,
            double reciprocalRank,
            long latencyMicros,
            List<RankedResult> results) {
    }

    private record RankedResult(String id, String instruction, float score) {
    }

    private record DedupMetrics(
            float threshold,
            int labeledPairCount,
            int truePositive,
            int falsePositive,
            int trueNegative,
            int falseNegative,
            double precision,
            double recall,
            double f1,
            ThresholdScore calibratedBest,
            List<ThresholdScore> thresholdSweep,
            List<PairError> errors) {
    }

    private record ThresholdScore(
            float threshold,
            int truePositive,
            int falsePositive,
            int trueNegative,
            int falseNegative,
            double precision,
            double recall,
            double f1) {
    }

    private record PairError(
            String errorType,
            String label,
            String leftId,
            String leftInstruction,
            String rightId,
            String rightInstruction,
            float similarity) {
    }
}
