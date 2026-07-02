package com.yikeyang.agenttrace.model;

public record SearchRequest(
        float[] embedding,
        Integer k,
        String platform,
        String app,
        Boolean success) {

    public int requestedK() {
        return k == null ? 5 : k;
    }
}
