package com.yikeyang.agenttrace.evaluation;

import java.util.List;

public record EvaluationLabels(
        String name,
        String version,
        String annotationPolicy,
        List<LabelGroup> retrievalGroups,
        List<LabelGroup> duplicateGroups,
        List<NegativePair> negativePairs) {

    public EvaluationLabels {
        retrievalGroups = retrievalGroups == null ? List.of() : List.copyOf(retrievalGroups);
        duplicateGroups = duplicateGroups == null ? List.of() : List.copyOf(duplicateGroups);
        negativePairs = negativePairs == null ? List.of() : List.copyOf(negativePairs);
    }

    public record LabelGroup(String label, String rationale, List<String> memberIds) {

        public LabelGroup {
            memberIds = memberIds == null ? List.of() : List.copyOf(memberIds);
        }
    }

    public record NegativePair(
            String leftId,
            String rightId,
            String rationale) {
    }
}
