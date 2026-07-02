package com.yikeyang.agenttrace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.embedding.MiniLmOnnxEmbedding;
import com.yikeyang.agenttrace.embedding.TrajectoryText;
import com.yikeyang.agenttrace.io.TrajectoryFileLoader;
import com.yikeyang.agenttrace.model.Trajectory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

public final class ReembedApplication {

    private ReembedApplication() {
    }

    public static void run(String[] args) throws Exception {
        Map<String, String> options = parseOptions(args);
        Path inputPath = Path.of(options.getOrDefault(
                "input", "sample-data/aguvis-500.json"));
        Path outputPath = Path.of(options.getOrDefault(
                "output", "sample-data/aguvis-500-minilm.json"));
        Path modelPath = Path.of(options.getOrDefault(
                "model",
                "models/all-MiniLM-L6-v2/model_qint8_arm64.onnx"));
        Path vocabularyPath = Path.of(options.getOrDefault(
                "vocab", "models/all-MiniLM-L6-v2/vocab.txt"));
        int maxLength = Integer.parseInt(
                options.getOrDefault("max-length", "256"));
        int batchSize = Integer.parseInt(
                options.getOrDefault("batch-size", "16"));

        ObjectMapper objectMapper = new ObjectMapper();
        List<Trajectory> trajectories =
                new TrajectoryFileLoader(objectMapper).load(inputPath);
        List<String> texts = trajectories.stream().map(TrajectoryText::from).toList();

        long started = System.nanoTime();
        String modelName;
        List<float[]> embeddings;
        try (MiniLmOnnxEmbedding model = new MiniLmOnnxEmbedding(
                modelPath, vocabularyPath, maxLength, batchSize)) {
            modelName = model.name();
            embeddings = model.embedBatch(texts);
        }
        long embeddingMillis = Math.round(
                (System.nanoTime() - started) / 1_000_000.0);

        List<Trajectory> updated = new ArrayList<>(trajectories.size());
        for (int i = 0; i < trajectories.size(); i++) {
            Trajectory original = trajectories.get(i);
            updated.add(new Trajectory(
                    original.id(),
                    original.instruction(),
                    original.platform(),
                    original.app(),
                    original.success(),
                    original.actions(),
                    embeddings.get(i),
                    original.source(),
                    original.sourceId(),
                    original.imageCount()));
        }

        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        objectMapper.writerWithDefaultPrettyPrinter()
                .writeValue(outputPath.toFile(), updated);
        double trajectoriesPerSecond =
                updated.size() * 1000.0 / Math.max(embeddingMillis, 1);
        if (options.containsKey("report")) {
            Path reportPath = Path.of(options.get("report"));
            Path reportParent = reportPath.toAbsolutePath().getParent();
            if (reportParent != null) {
                Files.createDirectories(reportParent);
            }
            EmbeddingRunReport report = new EmbeddingRunReport(
                    Instant.now().toString(),
                    inputPath.toString(),
                    outputPath.toString(),
                    modelName,
                    modelPath.toString(),
                    sha256(modelPath),
                    vocabularyPath.toString(),
                    updated.size(),
                    MiniLmOnnxEmbedding.DIMENSION,
                    maxLength,
                    batchSize,
                    embeddingMillis,
                    trajectoriesPerSecond,
                    System.getProperty("java.version"),
                    System.getProperty("os.name"),
                    System.getProperty("os.arch"));
            objectMapper.writerWithDefaultPrettyPrinter()
                    .writeValue(reportPath.toFile(), report);
            System.out.println("Wrote embedding report to " + reportPath);
        }
        System.out.printf(
                "Embedded %d trajectories with %s to %s "
                        + "(%d dimensions, %d ms, %.2f trajectories/s)%n",
                updated.size(),
                modelName,
                outputPath,
                MiniLmOnnxEmbedding.DIMENSION,
                embeddingMillis,
                trajectoriesPerSecond);
    }

    private static String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (var input = Files.newInputStream(path)) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        }
        return HexFormat.of().formatHex(digest.digest());
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

    private record EmbeddingRunReport(
            String generatedAt,
            String inputPath,
            String outputPath,
            String model,
            String modelPath,
            String modelSha256,
            String vocabularyPath,
            int trajectoryCount,
            int embeddingDimension,
            int maxSequenceLength,
            int batchSize,
            long embeddingMillis,
            double trajectoriesPerSecond,
            String javaVersion,
            String operatingSystem,
            String architecture) {
    }
}
