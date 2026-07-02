package com.yikeyang.agenttrace.model;

import java.util.List;

public record DuplicateGroup(
        String canonicalId,
        List<String> memberIds,
        float meanCosineSimilarity) {
}

