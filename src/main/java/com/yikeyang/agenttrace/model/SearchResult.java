package com.yikeyang.agenttrace.model;

import java.util.List;

public record SearchResult(
        String id,
        String instruction,
        String platform,
        String app,
        boolean success,
        List<String> actions,
        String source,
        String sourceId,
        int imageCount,
        float score) {
}
