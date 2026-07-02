package com.yikeyang.agenttrace.model;

public record SearchByTrajectoryRequest(
        String trajectoryId,
        Integer k,
        Boolean success) {

    public int requestedK() {
        return k == null ? 5 : k;
    }
}

