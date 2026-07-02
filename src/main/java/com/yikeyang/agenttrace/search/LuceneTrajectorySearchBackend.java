package com.yikeyang.agenttrace.search;

import com.yikeyang.agenttrace.model.DuplicateGroup;
import com.yikeyang.agenttrace.model.IndexStats;
import com.yikeyang.agenttrace.model.SearchRequest;
import com.yikeyang.agenttrace.model.SearchResult;
import com.yikeyang.agenttrace.model.Trajectory;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public final class LuceneTrajectorySearchBackend implements TrajectorySearchBackend {

    private static final String ID = "id";
    private static final String INSTRUCTION = "instruction";
    private static final String PLATFORM = "platform";
    private static final String APP = "app";
    private static final String SUCCESS = "success";
    private static final String ACTIONS = "actions";
    private static final String SOURCE = "source";
    private static final String SOURCE_ID = "source_id";
    private static final String IMAGE_COUNT = "image_count";
    private static final String EMBEDDING = "embedding";
    private static final String ACTION_SEPARATOR = "\u001F";

    private final Directory directory;
    private final IndexWriter writer;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private DirectoryReader reader;
    private volatile IndexStats stats = new IndexStats(0, 0, "lucene-hnsw");

    public LuceneTrajectorySearchBackend(Path indexPath) throws IOException {
        Files.createDirectories(indexPath);
        directory = FSDirectory.open(indexPath);
        writer = new IndexWriter(directory, new IndexWriterConfig());
    }

    @Override
    public void rebuild(List<Trajectory> trajectories) throws IOException {
        if (trajectories == null || trajectories.isEmpty()) {
            throw new IllegalArgumentException("at least one trajectory is required");
        }
        int dimension = trajectories.getFirst().embedding().length;
        Set<String> ids = new HashSet<>();
        for (Trajectory trajectory : trajectories) {
            if (trajectory.embedding().length != dimension) {
                throw new IllegalArgumentException("all embeddings must have dimension " + dimension);
            }
            if (!ids.add(trajectory.id())) {
                throw new IllegalArgumentException("duplicate trajectory id: " + trajectory.id());
            }
        }

        lock.writeLock().lock();
        try {
            writer.deleteAll();
            for (Trajectory trajectory : trajectories) {
                writer.addDocument(toDocument(trajectory));
            }
            writer.commit();
            refreshReader();
            stats = new IndexStats(trajectories.size(), dimension, "lucene-hnsw");
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public List<SearchResult> search(SearchRequest request) throws IOException {
        validateSearchRequest(request);
        lock.readLock().lock();
        try {
            if (reader == null) {
                return List.of();
            }
            Query filter = buildFilter(request);
            Query vectorQuery = filter == null
                    ? new KnnFloatVectorQuery(EMBEDDING, request.embedding(), request.requestedK())
                    : new KnnFloatVectorQuery(
                            EMBEDDING, request.embedding(), request.requestedK(), filter);
            IndexSearcher searcher = new IndexSearcher(reader);
            TopDocs topDocs = searcher.search(vectorQuery, request.requestedK());
            List<SearchResult> results = new ArrayList<>(topDocs.scoreDocs.length);
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document document = searcher.storedFields().document(scoreDoc.doc);
                results.add(fromDocument(document, scoreDoc.score));
            }
            return List.copyOf(results);
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public List<DuplicateGroup> findDuplicateGroups(
            List<Trajectory> trajectories, float threshold, int candidateK) throws IOException {
        if (threshold < -1.0f || threshold > 1.0f) {
            throw new IllegalArgumentException("threshold must be between -1 and 1");
        }
        if (candidateK < 2 || candidateK > 100) {
            throw new IllegalArgumentException("candidateK must be between 2 and 100");
        }
        if (trajectories == null || trajectories.isEmpty()) {
            return List.of();
        }

        Map<String, Trajectory> byId = new HashMap<>();
        trajectories.forEach(trajectory -> byId.put(trajectory.id(), trajectory));
        UnionFind unionFind = new UnionFind(trajectories.stream().map(Trajectory::id).toList());

        for (Trajectory trajectory : trajectories) {
            SearchRequest request = new SearchRequest(
                    trajectory.embedding(),
                    Math.min(candidateK, trajectories.size()),
                    trajectory.platform(),
                    trajectory.app(),
                    trajectory.success());
            for (SearchResult candidate : search(request)) {
                if (trajectory.id().equals(candidate.id())) {
                    continue;
                }
                Trajectory candidateTrajectory = byId.get(candidate.id());
                if (candidateTrajectory == null) {
                    continue;
                }
                float similarity = VectorMath.cosineSimilarity(
                        trajectory.embedding(), candidateTrajectory.embedding());
                if (similarity >= threshold) {
                    unionFind.union(trajectory.id(), candidate.id());
                }
            }
        }

        Map<String, List<String>> groups = new LinkedHashMap<>();
        trajectories.stream()
                .map(Trajectory::id)
                .sorted()
                .forEach(id -> groups.computeIfAbsent(unionFind.find(id), ignored -> new ArrayList<>())
                        .add(id));

        return groups.values().stream()
                .filter(group -> group.size() > 1)
                .map(group -> toDuplicateGroup(group, byId))
                .sorted(Comparator.comparing(DuplicateGroup::canonicalId))
                .toList();
    }

    @Override
    public IndexStats stats() {
        return stats;
    }

    @Override
    public void close() throws IOException {
        lock.writeLock().lock();
        try {
            if (reader != null) {
                reader.close();
                reader = null;
            }
            writer.close();
            directory.close();
        } finally {
            lock.writeLock().unlock();
        }
    }

    private Document toDocument(Trajectory trajectory) {
        Document document = new Document();
        document.add(new StringField(ID, trajectory.id(), Field.Store.YES));
        document.add(new StoredField(INSTRUCTION, trajectory.instruction()));
        document.add(new StringField(PLATFORM, trajectory.platform(), Field.Store.YES));
        document.add(new StringField(APP, trajectory.app(), Field.Store.YES));
        document.add(new StringField(SUCCESS, Boolean.toString(trajectory.success()), Field.Store.YES));
        document.add(new StoredField(ACTIONS, String.join(ACTION_SEPARATOR, trajectory.actions())));
        document.add(new StoredField(SOURCE, trajectory.source()));
        document.add(new StoredField(SOURCE_ID, trajectory.sourceId()));
        document.add(new StoredField(IMAGE_COUNT, trajectory.imageCount()));
        document.add(new KnnFloatVectorField(
                EMBEDDING, trajectory.embedding(), VectorSimilarityFunction.COSINE));
        return document;
    }

    private SearchResult fromDocument(Document document, float score) {
        String storedActions = document.get(ACTIONS);
        List<String> actions = storedActions == null || storedActions.isEmpty()
                ? List.of()
                : List.of(storedActions.split(ACTION_SEPARATOR, -1));
        return new SearchResult(
                document.get(ID),
                document.get(INSTRUCTION),
                document.get(PLATFORM),
                document.get(APP),
                Boolean.parseBoolean(document.get(SUCCESS)),
                actions,
                document.get(SOURCE),
                document.get(SOURCE_ID),
                document.getField(IMAGE_COUNT).numericValue().intValue(),
                score);
    }

    private Query buildFilter(SearchRequest request) {
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        boolean hasFilter = false;
        if (request.platform() != null && !request.platform().isBlank()) {
            builder.add(
                    new TermQuery(new Term(PLATFORM, request.platform())),
                    BooleanClause.Occur.FILTER);
            hasFilter = true;
        }
        if (request.app() != null && !request.app().isBlank()) {
            builder.add(
                    new TermQuery(new Term(APP, request.app())),
                    BooleanClause.Occur.FILTER);
            hasFilter = true;
        }
        if (request.success() != null) {
            builder.add(
                    new TermQuery(new Term(SUCCESS, request.success().toString())),
                    BooleanClause.Occur.FILTER);
            hasFilter = true;
        }
        return hasFilter ? builder.build() : null;
    }

    private void validateSearchRequest(SearchRequest request) {
        if (request == null || request.embedding() == null) {
            throw new IllegalArgumentException("embedding is required");
        }
        if (request.embedding().length != stats.vectorDimension()) {
            throw new IllegalArgumentException(
                    "embedding dimension must be " + stats.vectorDimension());
        }
        if (request.requestedK() < 1 || request.requestedK() > 100) {
            throw new IllegalArgumentException("k must be between 1 and 100");
        }
    }

    private void refreshReader() throws IOException {
        DirectoryReader nextReader = reader == null
                ? DirectoryReader.open(writer)
                : DirectoryReader.openIfChanged(reader, writer);
        if (nextReader != null && nextReader != reader) {
            DirectoryReader previous = reader;
            reader = nextReader;
            if (previous != null) {
                previous.close();
            }
        }
    }

    private DuplicateGroup toDuplicateGroup(
            List<String> memberIds, Map<String, Trajectory> byId) {
        List<String> sorted = memberIds.stream().sorted().toList();
        double total = 0.0;
        int pairs = 0;
        for (int i = 0; i < sorted.size(); i++) {
            for (int j = i + 1; j < sorted.size(); j++) {
                total += VectorMath.cosineSimilarity(
                        byId.get(sorted.get(i)).embedding(),
                        byId.get(sorted.get(j)).embedding());
                pairs++;
            }
        }
        float mean = pairs == 0 ? 1.0f : (float) (total / pairs);
        return new DuplicateGroup(sorted.getFirst(), sorted, mean);
    }

    private static final class UnionFind {
        private final Map<String, String> parent = new HashMap<>();

        private UnionFind(List<String> ids) {
            ids.forEach(id -> parent.put(id, id));
        }

        private String find(String id) {
            String root = parent.get(id);
            if (root == null) {
                throw new IllegalArgumentException("unknown id: " + id);
            }
            if (!root.equals(id)) {
                parent.put(id, find(root));
            }
            return parent.get(id);
        }

        private void union(String left, String right) {
            String leftRoot = find(left);
            String rightRoot = find(right);
            if (!leftRoot.equals(rightRoot)) {
                String first = leftRoot.compareTo(rightRoot) <= 0 ? leftRoot : rightRoot;
                String second = first.equals(leftRoot) ? rightRoot : leftRoot;
                parent.put(second, first);
            }
        }
    }
}
