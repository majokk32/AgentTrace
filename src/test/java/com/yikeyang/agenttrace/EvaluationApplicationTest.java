package com.yikeyang.agenttrace;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.evaluation.EvaluationLabels;
import com.yikeyang.agenttrace.evaluation.EvaluationLabels.LabelGroup;
import com.yikeyang.agenttrace.model.Trajectory;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class EvaluationApplicationTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void evaluatesRetrievalAndCalibratesDeduplication() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        Path data = temporaryDirectory.resolve("data.json");
        Path labels = temporaryDirectory.resolve("labels.json");
        Path report = temporaryDirectory.resolve("report.json");
        mapper.writeValue(data.toFile(), List.of(
                trajectory("a1", "alpha one", new float[] {1.0f, 0.0f}),
                trajectory("a2", "alpha two", new float[] {0.99f, 0.1f}),
                trajectory("b1", "beta one", new float[] {0.0f, 1.0f}),
                trajectory("b2", "beta two", new float[] {0.1f, 0.99f})));
        LabelGroup alpha = new LabelGroup(
                "alpha", "same alpha intent", List.of("a1", "a2"));
        LabelGroup beta = new LabelGroup(
                "beta", "same beta intent", List.of("b1", "b2"));
        mapper.writeValue(labels.toFile(), new EvaluationLabels(
                "test labels",
                "1",
                "test policy",
                List.of(alpha, beta),
                List.of(alpha, beta),
                List.of()));

        EvaluationApplication.run(new String[] {
                "--data", data.toString(),
                "--labels", labels.toString(),
                "--index", temporaryDirectory.resolve("index").toString(),
                "--output", report.toString(),
                "--embedding-model", "test",
                "--k", "1",
                "--duplicate-threshold", "0.8"
        });

        JsonNode result = mapper.readTree(report.toFile());
        assertEquals(System.getProperty("os.name"),
                result.path("operatingSystem").asText());
        assertEquals(1.0, result.path("retrieval").path("hitRateAtK").asDouble());
        assertEquals(1.0, result.path("retrieval")
                .path("meanReciprocalRank").asDouble());
        assertEquals(1.0, result.path("deduplication")
                .path("calibratedBest").path("f1").asDouble());
    }

    private static Trajectory trajectory(
            String id, String instruction, float[] embedding) {
        return new Trajectory(
                id,
                instruction,
                "mobile",
                "test",
                true,
                List.of("tap"),
                embedding);
    }
}
