package com.yikeyang.agenttrace.importer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.embedding.FeatureHashEmbedding;
import com.yikeyang.agenttrace.model.Trajectory;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class AguvisRowParser {

    private final ObjectMapper objectMapper;
    private final int embeddingDimension;

    public AguvisRowParser(ObjectMapper objectMapper, int embeddingDimension) {
        this.objectMapper = objectMapper;
        this.embeddingDimension = embeddingDimension;
    }

    public Trajectory parse(JsonNode rowWrapper) throws IOException {
        JsonNode row = rowWrapper.path("row");
        JsonNode messages = objectMapper.readTree(requiredText(row, "messages"));
        JsonNode metadata = objectMapper.readTree(requiredText(row, "metadata"));
        JsonNode others = metadata.path("others");

        String datasetId = textOrFallback(
                others.path("id"), "aguvis-row-" + rowWrapper.path("row_idx").asLong());
        String sourceId = textOrFallback(others.path("source_id"), datasetId);
        String source = textOrFallback(others.path("source"), "cua-lite/Aguvis");
        String instruction = extractInstruction(messages);
        List<String> actions = extractActions(messages);
        boolean success = extractSuccess(messages);
        String platform = textOrFallback(metadata.path("platform"), "mobile");
        String app = inferDatasetVariant(datasetId);
        int imageCount = row.path("images").isArray() ? row.path("images").size() : 0;
        float[] embedding = FeatureHashEmbedding.embed(
                instruction + " " + app, actions, embeddingDimension);

        return new Trajectory(
                datasetId,
                instruction,
                platform,
                app,
                success,
                actions,
                embedding,
                source,
                sourceId,
                imageCount);
    }

    private String extractInstruction(JsonNode messages) {
        for (JsonNode message : messages) {
            if (!"user".equals(message.path("role").asText())) {
                continue;
            }
            for (JsonNode content : message.path("content")) {
                if ("text".equals(content.path("type").asText())
                        && !content.path("text").asText().isBlank()) {
                    return content.path("text").asText().trim();
                }
            }
        }
        throw new IllegalArgumentException("AGUVIS row has no user instruction");
    }

    private List<String> extractActions(JsonNode messages) {
        List<String> actions = new ArrayList<>();
        for (JsonNode message : messages) {
            if (!"assistant".equals(message.path("role").asText())) {
                continue;
            }
            String description = extractActionDescription(message.path("content"));
            boolean addedTool = false;
            for (JsonNode toolCall : message.path("tool_calls")) {
                JsonNode function = toolCall.path("function");
                String name = function.path("name").asText();
                if (name.isBlank() || "terminate".equals(name)) {
                    continue;
                }
                actions.add(description.isBlank() ? name : name + ": " + description);
                addedTool = true;
            }
            if (!addedTool && !description.isBlank()) {
                actions.add(description);
            }
        }
        return List.copyOf(actions);
    }

    private String extractActionDescription(JsonNode content) {
        for (JsonNode item : content) {
            if ("action_description".equals(item.path("type").asText())) {
                return item.path("text").asText("").trim();
            }
        }
        return "";
    }

    private boolean extractSuccess(JsonNode messages) {
        Boolean terminalStatus = null;
        for (JsonNode message : messages) {
            for (JsonNode toolCall : message.path("tool_calls")) {
                JsonNode function = toolCall.path("function");
                if (!"terminate".equals(function.path("name").asText())) {
                    continue;
                }
                String status = function.path("arguments").path("status")
                        .asText()
                        .toLowerCase(Locale.ROOT);
                terminalStatus = "success".equals(status);
            }
        }
        // Demonstration trajectories without an explicit terminal marker are
        // considered successful unless the source says otherwise.
        return terminalStatus == null || terminalStatus;
    }

    private String inferDatasetVariant(String id) {
        String normalized = id.toLowerCase(Locale.ROOT);
        if (normalized.contains("android_control")) {
            return "android-control";
        }
        if (normalized.contains("aitw")) {
            return "aitw";
        }
        if (normalized.contains("coat")) {
            return "coat";
        }
        if (normalized.contains("guide")) {
            return "guide";
        }
        return "mobile-gui";
    }

    private static String requiredText(JsonNode row, String field) {
        String value = row.path(field).asText();
        if (value.isBlank()) {
            throw new IllegalArgumentException("AGUVIS row is missing " + field);
        }
        return value;
    }

    private static String textOrFallback(JsonNode value, String fallback) {
        String text = value.asText();
        return text.isBlank() ? fallback : text;
    }
}

