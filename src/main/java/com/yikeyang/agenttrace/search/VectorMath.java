package com.yikeyang.agenttrace.search;

public final class VectorMath {

    private VectorMath() {
    }

    public static float cosineSimilarity(float[] left, float[] right) {
        if (left == null || right == null || left.length != right.length || left.length == 0) {
            throw new IllegalArgumentException("vectors must be non-empty and have equal dimensions");
        }

        double dot = 0.0;
        double leftNorm = 0.0;
        double rightNorm = 0.0;
        for (int i = 0; i < left.length; i++) {
            dot += (double) left[i] * right[i];
            leftNorm += (double) left[i] * left[i];
            rightNorm += (double) right[i] * right[i];
        }

        if (leftNorm == 0.0 || rightNorm == 0.0) {
            return 0.0f;
        }
        return (float) (dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm)));
    }
}

