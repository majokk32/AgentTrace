package com.yikeyang.agenttrace.embedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Dependency-free baseline embedding for validating the data and search pipeline.
 *
 * <p>This is deliberately not presented as a semantic ML embedding. It hashes
 * instruction and action tokens into a fixed-size normalized vector so the
 * Lucene pipeline can be exercised on a storage-constrained development machine.
 */
public final class FeatureHashEmbedding {

    private FeatureHashEmbedding() {
    }

    public static float[] embed(String instruction, List<String> actions, int dimension) {
        if (dimension < 32 || dimension > 4096) {
            throw new IllegalArgumentException("dimension must be between 32 and 4096");
        }
        float[] vector = new float[dimension];
        addText(vector, instruction, 2.0f, "instruction");
        for (String action : actions) {
            addText(vector, action, 1.0f, "action");
        }

        double norm = 0.0;
        for (float value : vector) {
            norm += (double) value * value;
        }
        if (norm == 0.0) {
            vector[0] = 1.0f;
            return vector;
        }
        float divisor = (float) Math.sqrt(norm);
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= divisor;
        }
        return vector;
    }

    private static void addText(
            float[] vector, String text, float weight, String namespace) {
        if (text == null || text.isBlank()) {
            return;
        }
        List<String> tokens = tokenize(text);
        for (String token : tokens) {
            addFeature(vector, namespace + ":" + token, weight);
        }
        for (int i = 0; i + 1 < tokens.size(); i++) {
            addFeature(
                    vector,
                    namespace + ":bigram:" + tokens.get(i) + "_" + tokens.get(i + 1),
                    weight * 0.5f);
        }
    }

    private static List<String> tokenize(String text) {
        String[] rawTokens = text.toLowerCase(Locale.ROOT)
                .split("[^\\p{L}\\p{N}]+");
        List<String> tokens = new ArrayList<>(rawTokens.length);
        for (String token : rawTokens) {
            if (!token.isBlank()) {
                tokens.add(token);
            }
        }
        return tokens;
    }

    private static void addFeature(float[] vector, String feature, float weight) {
        long hash = fnv1a64(feature);
        int index = Math.floorMod(hash, vector.length);
        float sign = (hash & (1L << 63)) == 0 ? 1.0f : -1.0f;
        vector[index] += sign * weight;
    }

    private static long fnv1a64(String value) {
        long hash = 0xcbf29ce484222325L;
        for (int i = 0; i < value.length(); i++) {
            hash ^= value.charAt(i);
            hash *= 0x100000001b3L;
        }
        return hash;
    }
}

