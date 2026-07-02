package com.yikeyang.agenttrace.embedding;

import java.util.List;

public interface TextEmbeddingModel extends AutoCloseable {

    String name();

    int dimension();

    List<float[]> embedBatch(List<String> texts) throws Exception;

    default float[] embed(String text) throws Exception {
        return embedBatch(List.of(text)).getFirst();
    }

    @Override
    default void close() throws Exception {
    }
}
