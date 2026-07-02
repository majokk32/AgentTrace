package com.yikeyang.agenttrace.model;

public record DeduplicationRequest(Float threshold, Integer candidateK) {

    public float requestedThreshold() {
        return threshold == null ? 0.985f : threshold;
    }

    public int requestedCandidateK() {
        return candidateK == null ? 10 : candidateK;
    }
}

