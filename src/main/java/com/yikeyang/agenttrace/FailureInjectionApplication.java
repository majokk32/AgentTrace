package com.yikeyang.agenttrace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.embedding.MiniLmOnnxEmbedding;
import com.yikeyang.agenttrace.embedding.TrajectoryText;
import com.yikeyang.agenttrace.evaluation.EvaluationLabels;
import com.yikeyang.agenttrace.evaluation.EvaluationLabels.LabelGroup;
import com.yikeyang.agenttrace.evaluation.FailurePairSet;
import com.yikeyang.agenttrace.evaluation.FailurePairSet.FailurePair;
import com.yikeyang.agenttrace.io.TrajectoryFileLoader;
import com.yikeyang.agenttrace.model.Trajectory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class FailureInjectionApplication {

    private static final String FAILURE_SOURCE =
            "AgentTrace/fault-injection-truncated-actions-v1";

    private FailureInjectionApplication() {
    }

    public static void run(String[] args) throws Exception {
        Map<String, String> options = parseOptions(args);
        Path inputPath = Path.of(options.getOrDefault(
                "input", "sample-data/aguvis-500-minilm.json"));
        Path outputPath = Path.of(options.getOrDefault(
                "output", "sample-data/aguvis-500-plus-50-failures.json"));
        Path pairsPath = Path.of(options.getOrDefault(
                "pairs", "labels/aguvis-failure-pairs-50.json"));
        Path modelPath = Path.of(options.getOrDefault(
                "model", "models/all-MiniLM-L6-v2/model_qint8_arm64.onnx"));
        Path vocabularyPath = Path.of(options.getOrDefault(
                "vocab", "models/all-MiniLM-L6-v2/vocab.txt"));
        int count = Integer.parseInt(options.getOrDefault("count", "50"));
        int maxLength = Integer.parseInt(
                options.getOrDefault("max-length", "256"));
        int batchSize = Integer.parseInt(
                options.getOrDefault("batch-size", "16"));
        if (count < 1 || count > 500) {
            throw new IllegalArgumentException("count must be between 1 and 500");
        }

        ObjectMapper objectMapper = new ObjectMapper();
        List<Trajectory> originals =
                new TrajectoryFileLoader(objectMapper).load(inputPath);
        Map<String, Trajectory> originalsById = new LinkedHashMap<>();
        for (Trajectory original : originals) {
            originalsById.put(original.id(), original);
        }
        List<Trajectory> eligible = originals.stream()
                .filter(trajectory -> trajectory.success()
                        && trajectory.actions().size() >= 3)
                .toList();
        List<Trajectory> selected;
        Map<String, List<String>> alternativesById = new LinkedHashMap<>();
        String generationPolicy;
        if (options.containsKey("labels")) {
            Path labelsPath = Path.of(options.get("labels"));
            EvaluationLabels labels =
                    objectMapper.readValue(labelsPath.toFile(), EvaluationLabels.class);
            for (LabelGroup group : labels.retrievalGroups()) {
                for (String id : group.memberIds()) {
                    if (alternativesById.containsKey(id)) {
                        throw new IllegalArgumentException(
                                "retrieval label id appears in multiple groups: " + id);
                    }
                    alternativesById.put(
                            id,
                            group.memberIds().stream()
                                    .filter(candidate -> !candidate.equals(id))
                                    .toList());
                }
            }
            selected = alternativesById.keySet().stream()
                    .map(id -> {
                        Trajectory trajectory = originalsById.get(id);
                        if (trajectory == null) {
                            throw new IllegalArgumentException(
                                    "labels reference missing trajectory " + id);
                        }
                        if (!trajectory.success() || trajectory.actions().size() < 3) {
                            throw new IllegalArgumentException(
                                    "labeled trajectory is not eligible for truncation " + id);
                        }
                        return trajectory;
                    })
                    .toList();
            generationPolicy = "Create one controlled truncation failure for every "
                    + "trajectory in the retrieval label set. Keep the first two "
                    + "thirds of actions and record all other members of the same "
                    + "intent group as alternative successful targets.";
        } else {
            if (eligible.size() < count) {
                throw new IllegalArgumentException(
                        "only " + eligible.size() + " trajectories are eligible");
            }
            selected = selectEvenly(eligible, count);
            generationPolicy = "Deterministically select successful trajectories "
                    + "across the dataset and keep only the first two thirds of "
                    + "actions. Failures retain the original task text.";
        }

        List<FailureDraft> drafts = new ArrayList<>(selected.size());
        for (Trajectory original : selected) {
            int remainingActions = Math.max(
                    1, Math.min(
                            original.actions().size() - 1,
                            (original.actions().size() * 2) / 3));
            List<String> truncatedActions =
                    original.actions().subList(0, remainingActions);
            drafts.add(new FailureDraft(
                    original,
                    List.copyOf(truncatedActions),
                    original.id() + "-failed-truncated"));
        }

        long embeddingStarted = System.nanoTime();
        List<float[]> embeddings;
        try (MiniLmOnnxEmbedding model = new MiniLmOnnxEmbedding(
                modelPath, vocabularyPath, maxLength, batchSize)) {
            embeddings = model.embedBatch(drafts.stream()
                    .map(draft -> TrajectoryText.from(
                            draft.original().instruction(),
                            draft.original().app(),
                            draft.actions()))
                    .toList());
        }
        long embeddingMillis = Math.round(
                (System.nanoTime() - embeddingStarted) / 1_000_000.0);

        List<Trajectory> combined = new ArrayList<>(originals);
        List<FailurePair> pairs = new ArrayList<>(count);
        for (int i = 0; i < drafts.size(); i++) {
            FailureDraft draft = drafts.get(i);
            Trajectory original = draft.original();
            combined.add(new Trajectory(
                    draft.failureId(),
                    original.instruction(),
                    original.platform(),
                    original.app(),
                    false,
                    draft.actions(),
                    embeddings.get(i),
                    FAILURE_SOURCE,
                    original.id(),
                    0));
            pairs.add(new FailurePair(
                    draft.failureId(),
                    original.id(),
                    alternativesById.getOrDefault(original.id(), List.of()),
                    "truncated-action-sequence",
                    original.actions().size(),
                    draft.actions().size()));
        }

        writeParent(outputPath);
        writeParent(pairsPath);
        objectMapper.writerWithDefaultPrettyPrinter()
                .writeValue(outputPath.toFile(), combined);
        FailurePairSet pairSet = new FailurePairSet(
                "AGUVIS controlled failed-to-successful trajectory pairs",
                "1.0",
                Instant.now().toString(),
                generationPolicy,
                pairs);
        objectMapper.writerWithDefaultPrettyPrinter()
                .writeValue(pairsPath.toFile(), pairSet);
        System.out.printf(
                "Injected %d controlled failures into %s: %d total trajectories, "
                        + "%d ms embedding time%n",
                pairs.size(), outputPath, combined.size(), embeddingMillis);
        System.out.println("Wrote recovery ground truth to " + pairsPath);
    }

    private static List<Trajectory> selectEvenly(
            List<Trajectory> eligible, int count) {
        List<Trajectory> selected = new ArrayList<>(count);
        Set<String> selectedIds = new HashSet<>();
        for (int i = 0; i < count; i++) {
            int index = (int) ((long) i * eligible.size() / count);
            Trajectory trajectory = eligible.get(index);
            if (!selectedIds.add(trajectory.id())) {
                throw new IllegalStateException(
                        "deterministic selection produced a duplicate");
            }
            selected.add(trajectory);
        }
        return selected;
    }

    private static void writeParent(Path path) throws Exception {
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
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

    private record FailureDraft(
            Trajectory original, List<String> actions, String failureId) {
    }
}
