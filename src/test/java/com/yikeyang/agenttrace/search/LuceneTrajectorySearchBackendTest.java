package com.yikeyang.agenttrace.search;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.yikeyang.agenttrace.model.DuplicateGroup;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LuceneTrajectorySearchBackendTest {

    @TempDir
    Path tempDirectory;

    @Test
    void searchesAndFiltersTrajectories() throws Exception {
        List<Trajectory> trajectories = fixtures();
        try (LuceneTrajectorySearchBackend backend =
                     new LuceneTrajectorySearchBackend(tempDirectory.resolve("index"))) {
            backend.rebuild(trajectories);

            List<SearchResult> results = backend.search(new SearchRequest(
                    new float[]{1.0f, 0.0f, 0.0f, 0.0f},
                    3,
                    "mobile",
                    "android-settings",
                    true));

            assertEquals("wifi-success-1", results.getFirst().id());
            assertTrue(results.stream().allMatch(SearchResult::success));
            assertTrue(results.stream().allMatch(result -> result.platform().equals("mobile")));
            assertEquals(4, backend.stats().vectorDimension());
        }
    }

    @Test
    void groupsNearDuplicateTrajectories() throws Exception {
        List<Trajectory> trajectories = fixtures();
        try (LuceneTrajectorySearchBackend backend =
                     new LuceneTrajectorySearchBackend(tempDirectory.resolve("index"))) {
            backend.rebuild(trajectories);

            List<DuplicateGroup> groups =
                    backend.findDuplicateGroups(trajectories, 0.995f, 4);

            assertFalse(groups.isEmpty());
            DuplicateGroup wifiGroup = groups.stream()
                    .filter(group -> group.memberIds().contains("wifi-success-1"))
                    .findFirst()
                    .orElseThrow();
            assertTrue(wifiGroup.memberIds().contains("wifi-success-2"));
            assertFalse(wifiGroup.memberIds().contains("wifi-failed"));
            assertTrue(wifiGroup.meanCosineSimilarity() >= 0.995f);
        }
    }

    private static List<Trajectory> fixtures() {
        return List.of(
                trajectory(
                        "wifi-success-1", "Turn on Wi-Fi", true,
                        new float[]{1.0f, 0.0f, 0.0f, 0.0f}),
                trajectory(
                        "wifi-success-2", "Enable wireless network", true,
                        new float[]{0.999f, 0.010f, 0.0f, 0.0f}),
                trajectory(
                        "wifi-failed", "Turn on Wi-Fi", false,
                        new float[]{0.990f, 0.020f, 0.0f, 0.0f}),
                trajectory(
                        "notification-success", "Disable notification previews", true,
                        new float[]{0.0f, 1.0f, 0.0f, 0.0f}),
                trajectory(
                        "login-failed", "Sign in to the account", false,
                        new float[]{0.0f, 0.0f, 1.0f, 0.0f}));
    }

    private static Trajectory trajectory(
            String id, String instruction, boolean success, float[] embedding) {
        return new Trajectory(
                id,
                instruction,
                "mobile",
                "android-settings",
                success,
                List.of("tap:settings", "tap:target"),
                embedding);
    }
}
