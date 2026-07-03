package com.yikeyang.agenttrace.search;

import com.yikeyang.agenttrace.model.DuplicateGroup;
import com.yikeyang.agenttrace.model.IndexStats;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public interface TrajectorySearchBackend extends AutoCloseable {

    void rebuild(List<Trajectory> trajectories) throws IOException;

    List<SearchResult> search(SearchRequest request) throws IOException;

    default List<List<SearchResult>> searchBatch(List<SearchRequest> requests)
            throws IOException {
        if (requests == null || requests.isEmpty()) {
            return List.of();
        }
        List<List<SearchResult>> results = new ArrayList<>(requests.size());
        for (SearchRequest request : requests) {
            results.add(search(request));
        }
        return List.copyOf(results);
    }

    List<DuplicateGroup> findDuplicateGroups(
            List<Trajectory> trajectories, float threshold, int candidateK) throws IOException;

    IndexStats stats();

    @Override
    void close() throws IOException;
}
