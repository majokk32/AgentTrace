package com.yikeyang.agenttrace.embedding;

import com.yikeyang.agenttrace.model.Trajectory;
import java.util.List;

public final class TrajectoryText {

    private TrajectoryText() {
    }

    public static String from(Trajectory trajectory) {
        return from(
                trajectory.instruction(),
                trajectory.app(),
                trajectory.actions());
    }

    public static String from(
            String instruction, String app, List<String> actions) {
        StringBuilder text = new StringBuilder()
                .append("Task: ")
                .append(instruction)
                .append(". App: ")
                .append(app)
                .append(". Steps: ");
        for (int i = 0; i < actions.size(); i++) {
            if (i > 0) {
                text.append(' ');
            }
            text.append(actions.get(i));
        }
        return text.toString();
    }
}
