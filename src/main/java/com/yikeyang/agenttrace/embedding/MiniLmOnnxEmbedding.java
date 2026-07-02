package com.yikeyang.agenttrace.embedding;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Local semantic embedding with all-MiniLM-L6-v2 and ONNX Runtime.
 */
public final class MiniLmOnnxEmbedding implements TextEmbeddingModel {

    public static final int DIMENSION = 384;

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final WordPieceTokenizer tokenizer;
    private final int batchSize;
    private final String modelName;

    public static Path defaultModelPath() {
        String architecture =
                System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        String fileName = architecture.contains("aarch64")
                || architecture.contains("arm64")
                ? "model_qint8_arm64.onnx"
                : "model_quint8_avx2.onnx";
        return Path.of("models", "all-MiniLM-L6-v2", fileName);
    }

    public MiniLmOnnxEmbedding(
            Path modelPath,
            Path vocabularyPath,
            int maxLength,
            int batchSize) throws Exception {
        if (!Files.isRegularFile(modelPath)) {
            throw new IllegalArgumentException("model file not found: " + modelPath);
        }
        if (!Files.isRegularFile(vocabularyPath)) {
            throw new IllegalArgumentException(
                    "vocabulary file not found: " + vocabularyPath);
        }
        if (batchSize < 1 || batchSize > 128) {
            throw new IllegalArgumentException("batchSize must be between 1 and 128");
        }
        this.environment = OrtEnvironment.getEnvironment();
        this.tokenizer = new WordPieceTokenizer(vocabularyPath, maxLength);
        this.batchSize = batchSize;
        this.modelName = "sentence-transformers/all-MiniLM-L6-v2/"
                + modelPath.getFileName();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(
                    OrtSession.SessionOptions.OptLevel.ALL_OPT);
            this.session = environment.createSession(modelPath.toString(), options);
        }
        requireInput("input_ids");
        requireInput("attention_mask");
        requireInput("token_type_ids");
    }

    @Override
    public String name() {
        return modelName;
    }

    @Override
    public int dimension() {
        return DIMENSION;
    }

    @Override
    public List<float[]> embedBatch(List<String> texts) throws OrtException {
        if (texts.isEmpty()) {
            return List.of();
        }
        List<float[]> embeddings = new ArrayList<>(texts.size());
        for (int start = 0; start < texts.size(); start += batchSize) {
            int end = Math.min(start + batchSize, texts.size());
            embeddings.addAll(runBatch(texts.subList(start, end)));
        }
        return List.copyOf(embeddings);
    }

    private List<float[]> runBatch(List<String> texts) throws OrtException {
        List<WordPieceTokenizer.Encoding> encodings =
                texts.stream().map(tokenizer::encode).toList();
        int sequenceLength = encodings.stream()
                .mapToInt(encoding -> encoding.inputIds().length)
                .max()
                .orElseThrow();

        long[][] inputIds = padded(encodings, sequenceLength, InputKind.IDS);
        long[][] attentionMasks = padded(
                encodings, sequenceLength, InputKind.ATTENTION_MASK);
        long[][] tokenTypeIds = padded(
                encodings, sequenceLength, InputKind.TOKEN_TYPE_IDS);

        try (OnnxTensor inputIdsTensor =
                     OnnxTensor.createTensor(environment, inputIds);
             OnnxTensor attentionMaskTensor =
                     OnnxTensor.createTensor(environment, attentionMasks);
             OnnxTensor tokenTypeIdsTensor =
                     OnnxTensor.createTensor(environment, tokenTypeIds)) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);
            inputs.put("token_type_ids", tokenTypeIdsTensor);
            try (OrtSession.Result result = session.run(inputs)) {
                OnnxValue output = result.get("last_hidden_state")
                        .orElseGet(() -> result.get(0));
                Object value = output.getValue();
                if (!(value instanceof float[][][] hiddenStates)) {
                    throw new IllegalStateException(
                            "expected rank-3 float output but received "
                                    + value.getClass().getTypeName());
                }
                return meanPoolAndNormalize(hiddenStates, attentionMasks);
            }
        }
    }

    private long[][] padded(
            List<WordPieceTokenizer.Encoding> encodings,
            int sequenceLength,
            InputKind kind) {
        long[][] batch = new long[encodings.size()][sequenceLength];
        if (kind == InputKind.IDS && tokenizer.paddingTokenId() != 0) {
            for (long[] row : batch) {
                java.util.Arrays.fill(row, tokenizer.paddingTokenId());
            }
        }
        for (int i = 0; i < encodings.size(); i++) {
            long[] values = switch (kind) {
                case IDS -> encodings.get(i).inputIds();
                case ATTENTION_MASK -> encodings.get(i).attentionMask();
                case TOKEN_TYPE_IDS -> encodings.get(i).tokenTypeIds();
            };
            System.arraycopy(values, 0, batch[i], 0, values.length);
        }
        return batch;
    }

    private static List<float[]> meanPoolAndNormalize(
            float[][][] hiddenStates, long[][] attentionMasks) {
        List<float[]> embeddings = new ArrayList<>(hiddenStates.length);
        for (int batch = 0; batch < hiddenStates.length; batch++) {
            if (hiddenStates[batch].length != attentionMasks[batch].length) {
                throw new IllegalStateException(
                        "model output sequence length does not match attention mask");
            }
            float[] embedding = new float[DIMENSION];
            int tokenCount = 0;
            for (int token = 0; token < hiddenStates[batch].length; token++) {
                if (attentionMasks[batch][token] == 0L) {
                    continue;
                }
                if (hiddenStates[batch][token].length != DIMENSION) {
                    throw new IllegalStateException(
                            "expected " + DIMENSION + " output dimensions");
                }
                tokenCount++;
                for (int dimension = 0; dimension < DIMENSION; dimension++) {
                    embedding[dimension] += hiddenStates[batch][token][dimension];
                }
            }
            if (tokenCount == 0) {
                throw new IllegalStateException("model returned no unmasked tokens");
            }
            double norm = 0.0;
            for (int dimension = 0; dimension < DIMENSION; dimension++) {
                embedding[dimension] /= tokenCount;
                norm += (double) embedding[dimension] * embedding[dimension];
            }
            float divisor = (float) Math.sqrt(norm);
            if (divisor == 0.0f) {
                throw new IllegalStateException("model returned a zero embedding");
            }
            for (int dimension = 0; dimension < DIMENSION; dimension++) {
                embedding[dimension] /= divisor;
            }
            embeddings.add(embedding);
        }
        return embeddings;
    }

    private void requireInput(String inputName) {
        if (!session.getInputNames().contains(inputName)) {
            throw new IllegalArgumentException(
                    "ONNX model is missing required input " + inputName
                            + "; found " + session.getInputNames());
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }

    private enum InputKind {
        IDS,
        ATTENTION_MASK,
        TOKEN_TYPE_IDS
    }
}
