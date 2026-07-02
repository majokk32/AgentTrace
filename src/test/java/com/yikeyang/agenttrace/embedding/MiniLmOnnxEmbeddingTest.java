package com.yikeyang.agenttrace.embedding;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class MiniLmOnnxEmbeddingTest {

    @Test
    void selectsModelForHostArchitecture() {
        String originalArchitecture = System.getProperty("os.arch");
        try {
            System.setProperty("os.arch", "aarch64");
            assertEquals(
                    Path.of(
                            "models",
                            "all-MiniLM-L6-v2",
                            "model_qint8_arm64.onnx"),
                    MiniLmOnnxEmbedding.defaultModelPath());

            System.setProperty("os.arch", "amd64");
            assertEquals(
                    Path.of(
                            "models",
                            "all-MiniLM-L6-v2",
                            "model_quint8_avx2.onnx"),
                    MiniLmOnnxEmbedding.defaultModelPath());
        } finally {
            if (originalArchitecture == null) {
                System.clearProperty("os.arch");
            } else {
                System.setProperty("os.arch", originalArchitecture);
            }
        }
    }
}
