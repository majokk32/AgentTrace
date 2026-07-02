package com.yikeyang.agenttrace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yikeyang.agenttrace.io.TrajectoryFileLoader;
import com.yikeyang.agenttrace.model.Trajectory;
import com.yikeyang.agenttrace.search.CuvsTrajectorySearchBackend;
import com.yikeyang.agenttrace.search.SearchBackendFactory;
import com.yikeyang.agenttrace.search.TrajectorySearchBackend;
import com.yikeyang.agenttrace.server.AgentTraceServer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

public final class AgentTraceApplication {

    private AgentTraceApplication() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && "import-aguvis".equals(args[0])) {
            AguvisImportApplication.run(Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        if (args.length > 0 && "report".equals(args[0])) {
            DatasetReportApplication.run(Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        if (args.length > 0 && "embed".equals(args[0])) {
            ReembedApplication.run(Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        if (args.length > 0 && "evaluate".equals(args[0])) {
            EvaluationApplication.run(Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        if (args.length > 0 && "inject-failures".equals(args[0])) {
            FailureInjectionApplication.run(Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        if (args.length > 0 && "evaluate-recovery".equals(args[0])) {
            RecoveryEvaluationApplication.run(Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        Map<String, String> options = parseOptions(args);
        Path dataPath = Path.of(options.getOrDefault(
                "data", "sample-data/trajectories.json"));
        Path indexPath = Path.of(options.getOrDefault(
                "index", "data/lucene-index"));
        int port = Integer.parseInt(options.getOrDefault("port", "8080"));
        String backendName = options.getOrDefault("backend", "lucene");
        String cuvsUrl = options.getOrDefault(
                "cuvs-url", CuvsTrajectorySearchBackend.DEFAULT_URL);

        ObjectMapper objectMapper = new ObjectMapper();
        List<Trajectory> trajectories =
                new TrajectoryFileLoader(objectMapper).load(dataPath);
        TrajectorySearchBackend backend = SearchBackendFactory.create(
                backendName, indexPath, cuvsUrl, objectMapper);
        backend.rebuild(trajectories);
        AgentTraceServer server =
                new AgentTraceServer(port, backend, trajectories, objectMapper);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.close();
            try {
                backend.close();
            } catch (Exception exception) {
                System.err.println("Failed to close backend: " + exception.getMessage());
            }
        }));

        server.start();
        System.out.printf(
                "AgentTrace started on http://localhost:%d with %d trajectories (%d dimensions)%n",
                port, backend.stats().trajectoryCount(), backend.stats().vectorDimension());
        new CountDownLatch(1).await();
    }

    private static Map<String, String> parseOptions(String[] args) {
        Map<String, String> options = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String argument = args[i];
            if (!argument.startsWith("--") || i + 1 >= args.length) {
                throw new IllegalArgumentException(
                        "arguments must use --name value syntax");
            }
            options.put(argument.substring(2), args[++i]);
        }
        return options;
    }
}
