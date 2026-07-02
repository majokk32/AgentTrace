package com.yikeyang.agenttrace.search;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class VectorMathTest {

    @Test
    void computesCosineSimilarity() {
        assertEquals(1.0f, VectorMath.cosineSimilarity(
                new float[]{1.0f, 2.0f}, new float[]{2.0f, 4.0f}), 0.0001f);
        assertEquals(0.0f, VectorMath.cosineSimilarity(
                new float[]{1.0f, 0.0f}, new float[]{0.0f, 1.0f}), 0.0001f);
        assertEquals(-1.0f, VectorMath.cosineSimilarity(
                new float[]{1.0f, 0.0f}, new float[]{-1.0f, 0.0f}), 0.0001f);
    }
}

