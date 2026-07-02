package com.yikeyang.agenttrace.embedding;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.yikeyang.agenttrace.search.VectorMath;
import java.util.List;
import org.junit.jupiter.api.Test;

class FeatureHashEmbeddingTest {

    @Test
    void givesRelatedInstructionsMoreOverlapThanUnrelatedInstructions() {
        float[] alarm = FeatureHashEmbedding.embed(
                "Set an alarm for 5 PM", List.of("tap: open clock", "tap: set alarm"), 256);
        float[] related = FeatureHashEmbedding.embed(
                "Set a morning alarm", List.of("tap: open clock", "tap: add alarm"), 256);
        float[] unrelated = FeatureHashEmbedding.embed(
                "Read the latest news from Jamaica",
                List.of("type: news in Jamaica", "tap: news tab"),
                256);

        assertTrue(
                VectorMath.cosineSimilarity(alarm, related)
                        > VectorMath.cosineSimilarity(alarm, unrelated));
    }
}

