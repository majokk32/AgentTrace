package com.yikeyang.agenttrace.model;

public record IndexStats(
        long trajectoryCount,
        int vectorDimension,
        String backend) {
}

