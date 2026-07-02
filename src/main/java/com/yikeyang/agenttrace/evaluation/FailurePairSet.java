package com.yikeyang.agenttrace.evaluation;

import java.util.List;

public record FailurePairSet(
        String name,
        String version,
        String generatedAt,
        String generationPolicy,
        List<FailurePair> pairs) {

    public FailurePairSet {
        pairs = pairs == null ? List.of() : List.copyOf(pairs);
    }

    public record FailurePair(
            String failureId,
            String successfulId,
            List<String> alternativeSuccessfulIds,
            String failureType,
            int originalActionCount,
            int remainingActionCount) {

        public FailurePair {
            alternativeSuccessfulIds = alternativeSuccessfulIds == null
                    ? List.of()
                    : List.copyOf(alternativeSuccessfulIds);
        }
    }
}
