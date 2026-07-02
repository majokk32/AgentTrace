package com.yikeyang.agenttrace.search;

import com.yikeyang.agenttrace.model.DuplicateGroup;
import com.yikeyang.agenttrace.model.IndexStats;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import java.io.IOException;
import java.util.List;

public interface TrajectorySearchBackend extends AutoCloseable {

    void rebuild(List<Trajectory> trajectories) throws IOException;

    List<SearchResult> search(SearchRequest request) throws IOException;

    List<DuplicateGroup> findDuplicateGroups(
            List<Trajectory> trajectories, float threshold, int candidateK) throws IOException;

    IndexStats stats();

    @Override
    void close() throws IOException;
}

