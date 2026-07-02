package com.yikeyang.agenttrace;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.evaluation.FailurePairSet;
import com.yikeyang.agenttrace.evaluation.FailurePairSet.FailurePair;
import com.yikeyang.agenttrace.model.Trajectory;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RecoveryEvaluationApplicationTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void excludesParentAndFindsAlternativeSuccessfulTrajectory() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        Path data = temporaryDirectory.resolve("data.json");
        Path pairs = temporaryDirectory.resolve("pairs.json");
        Path report = temporaryDirectory.resolve("report.json");
        mapper.writeValue(data.toFile(), List.of(
                trajectory("parent", true, "parent", new float[] {1.0f, 0.0f}),
                trajectory("alternative", true, "alternative", new float[] {0.9f, 0.1f}),
                trajectory("irrelevant", true, "irrelevant", new float[] {0.0f, 1.0f}),
                trajectory("failure", false, "parent", new float[] {1.0f, 0.05f})));
        mapper.writeValue(pairs.toFile(), new FailurePairSet(
                "test pairs",
                "1",
                "now",
                "test policy",
                List.of(new FailurePair(
                        "failure",
                        "parent",
                        List.of("alternative"),
                        "truncated",
                        3,
                        2))));

        RecoveryEvaluationApplication.run(new String[] {
                "--data", data.toString(),
                "--pairs", pairs.toString(),
                "--index", temporaryDirectory.resolve("index").toString(),
                "--output", report.toString(),
                "--k", "2"
        });

        JsonNode result = mapper.readTree(report.toFile());
        assertEquals(System.getProperty("os.name"),
                result.path("operatingSystem").asText());
        assertEquals(1.0, result.path("parentRecallAt1").asDouble());
        assertEquals(1.0, result.path("parentExcludedIntentHitRateAtK").asDouble());
        assertEquals(1.0, result.path("parentExcludedIntentMeanReciprocalRank").asDouble());
    }

    private static Trajectory trajectory(
            String id, boolean success, String sourceId, float[] embedding) {
        return new Trajectory(
                id,
                id,
                "mobile",
                "test",
                success,
                List.of("tap"),
                embedding,
                success ? "test" : "AgentTrace/fault-injection-truncated-actions-v1",
                sourceId,
                0);
    }
}
