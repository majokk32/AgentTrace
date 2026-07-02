package com.yikeyang.agenttrace.model;

import java.util.List;
import java.util.Objects;

public record Trajectory(
        String id,
        String instruction,
        String platform,
        String app,
        boolean success,
        List<String> actions,
        float[] embedding,
        String source,
        String sourceId,
        int imageCount) {

    public Trajectory {
        id = requireText(id, "id");
        instruction = requireText(instruction, "instruction");
        platform = requireText(platform, "platform");
        app = requireText(app, "app");
        actions = actions == null ? List.of() : List.copyOf(actions);
        Objects.requireNonNull(embedding, "embedding must not be null");
        if (embedding.length == 0) {
            throw new IllegalArgumentException("embedding must not be empty");
        }
        embedding = embedding.clone();
        source = source == null || source.isBlank() ? "local" : source;
        sourceId = sourceId == null || sourceId.isBlank() ? id : sourceId;
        if (imageCount < 0) {
            throw new IllegalArgumentException("imageCount must not be negative");
        }
    }

    public Trajectory(
            String id,
            String instruction,
            String platform,
            String app,
            boolean success,
            List<String> actions,
            float[] embedding) {
        this(
                id,
                instruction,
                platform,
                app,
                success,
                actions,
                embedding,
                "local",
                id,
                0);
    }

    @Override
    public float[] embedding() {
        return embedding.clone();
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }
}
