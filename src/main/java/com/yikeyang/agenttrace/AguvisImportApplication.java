package com.yikeyang.agenttrace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.importer.AguvisRowApiImporter;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public final class AguvisImportApplication {

    private AguvisImportApplication() {
    }

    public static void run(String[] args) throws Exception {
        Map<String, String> options = parseOptions(args);
        Path output = Path.of(options.getOrDefault(
                "output", "sample-data/aguvis-500.json"));
        String config = options.getOrDefault("config", "mobile.navigation");
        String split = options.getOrDefault("split", "train");
        int startOffset = Integer.parseInt(options.getOrDefault("start-offset", "0"));
        int limit = Integer.parseInt(options.getOrDefault("limit", "500"));
        int dimension = Integer.parseInt(options.getOrDefault("dimension", "256"));

        AguvisRowApiImporter.ImportSummary summary =
                new AguvisRowApiImporter(new ObjectMapper()).importRows(
                        config, split, startOffset, limit, dimension, output);
        System.out.printf(
                "Imported %d AGUVIS trajectories to %s (%d dimensions, %.2f MiB)%n",
                summary.trajectoryCount(),
                output,
                summary.embeddingDimension(),
                summary.outputBytes() / 1024.0 / 1024.0);
        System.out.println("Dataset variants: " + summary.variantCounts());
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
}

