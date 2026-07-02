package com.yikeyang.agenttrace.io;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.model.Trajectory;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public final class TrajectoryFileLoader {

    private final ObjectMapper objectMapper;

    public TrajectoryFileLoader(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public List<Trajectory> load(Path path) throws IOException {
        List<Trajectory> trajectories = objectMapper.readValue(
                path.toFile(), new TypeReference<List<Trajectory>>() {
                });
        if (trajectories.isEmpty()) {
            throw new IllegalArgumentException("trajectory file is empty: " + path);
        }
        return List.copyOf(trajectories);
    }
}

