package com.yikeyang.agenttrace.embedding;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Minimal BERT-compatible uncased WordPiece tokenizer.
 *
 * <p>The implementation performs text cleaning, lower-casing, accent removal,
 * punctuation/CJK splitting, greedy WordPiece tokenization, and special-token
 * insertion. It deliberately covers the inference path needed by MiniLM
 * without pulling a second native tokenizer runtime into the project.
 */
public final class WordPieceTokenizer {

    private static final int MAX_INPUT_CHARS_PER_WORD = 100;

    private final Map<String, Integer> vocabulary;
    private final int unknownTokenId;
    private final int classificationTokenId;
    private final int separatorTokenId;
    private final int paddingTokenId;
    private final int maxLength;

    public WordPieceTokenizer(Path vocabularyPath, int maxLength) throws IOException {
        if (maxLength < 8 || maxLength > 512) {
            throw new IllegalArgumentException("maxLength must be between 8 and 512");
        }
        List<String> tokens = Files.readAllLines(vocabularyPath, StandardCharsets.UTF_8);
        this.vocabulary = new HashMap<>(tokens.size());
        for (int i = 0; i < tokens.size(); i++) {
            vocabulary.put(tokens.get(i), i);
        }
        this.unknownTokenId = requiredToken("[UNK]");
        this.classificationTokenId = requiredToken("[CLS]");
        this.separatorTokenId = requiredToken("[SEP]");
        this.paddingTokenId = requiredToken("[PAD]");
        this.maxLength = maxLength;
    }

    public Encoding encode(String text) {
        List<Integer> tokenIds = new ArrayList<>();
        for (String token : basicTokenize(text == null ? "" : text)) {
            if (tokenIds.size() >= maxLength - 2) {
                break;
            }
            List<Integer> wordPieces = wordPieceTokenize(token);
            int remaining = maxLength - 2 - tokenIds.size();
            tokenIds.addAll(wordPieces.subList(0, Math.min(remaining, wordPieces.size())));
        }

        long[] inputIds = new long[tokenIds.size() + 2];
        long[] attentionMask = new long[inputIds.length];
        long[] tokenTypeIds = new long[inputIds.length];
        inputIds[0] = classificationTokenId;
        for (int i = 0; i < tokenIds.size(); i++) {
            inputIds[i + 1] = tokenIds.get(i);
        }
        inputIds[inputIds.length - 1] = separatorTokenId;
        java.util.Arrays.fill(attentionMask, 1L);
        return new Encoding(inputIds, attentionMask, tokenTypeIds);
    }

    public int paddingTokenId() {
        return paddingTokenId;
    }

    private List<Integer> wordPieceTokenize(String token) {
        if (token.codePointCount(0, token.length()) > MAX_INPUT_CHARS_PER_WORD) {
            return List.of(unknownTokenId);
        }
        List<Integer> pieces = new ArrayList<>();
        int start = 0;
        while (start < token.length()) {
            int end = token.length();
            Integer pieceId = null;
            int matchedEnd = -1;
            while (start < end) {
                String candidate = token.substring(start, end);
                if (start > 0) {
                    candidate = "##" + candidate;
                }
                pieceId = vocabulary.get(candidate);
                if (pieceId != null) {
                    matchedEnd = end;
                    break;
                }
                end = token.offsetByCodePoints(end, -1);
            }
            if (pieceId == null) {
                return List.of(unknownTokenId);
            }
            pieces.add(pieceId);
            start = matchedEnd;
        }
        return pieces;
    }

    private static List<String> basicTokenize(String text) {
        String normalized = Normalizer.normalize(
                        text.toLowerCase(Locale.ROOT), Normalizer.Form.NFD)
                .replaceAll("\\p{M}+", "");
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        normalized.codePoints().forEach(codePoint -> {
            if (isControl(codePoint)) {
                return;
            }
            if (Character.isWhitespace(codePoint)) {
                flush(current, tokens);
            } else if (isPunctuation(codePoint) || isCjk(codePoint)) {
                flush(current, tokens);
                tokens.add(new String(Character.toChars(codePoint)));
            } else {
                current.appendCodePoint(codePoint);
            }
        });
        flush(current, tokens);
        return tokens;
    }

    private static boolean isControl(int codePoint) {
        int type = Character.getType(codePoint);
        return type == Character.CONTROL || type == Character.FORMAT;
    }

    private static boolean isPunctuation(int codePoint) {
        int type = Character.getType(codePoint);
        return type == Character.CONNECTOR_PUNCTUATION
                || type == Character.DASH_PUNCTUATION
                || type == Character.START_PUNCTUATION
                || type == Character.END_PUNCTUATION
                || type == Character.INITIAL_QUOTE_PUNCTUATION
                || type == Character.FINAL_QUOTE_PUNCTUATION
                || type == Character.OTHER_PUNCTUATION;
    }

    private static boolean isCjk(int codePoint) {
        return (codePoint >= 0x4E00 && codePoint <= 0x9FFF)
                || (codePoint >= 0x3400 && codePoint <= 0x4DBF)
                || (codePoint >= 0x20000 && codePoint <= 0x2A6DF)
                || (codePoint >= 0x2A700 && codePoint <= 0x2B73F)
                || (codePoint >= 0x2B740 && codePoint <= 0x2B81F)
                || (codePoint >= 0x2B820 && codePoint <= 0x2CEAF)
                || (codePoint >= 0xF900 && codePoint <= 0xFAFF)
                || (codePoint >= 0x2F800 && codePoint <= 0x2FA1F);
    }

    private static void flush(StringBuilder current, List<String> tokens) {
        if (!current.isEmpty()) {
            tokens.add(current.toString());
            current.setLength(0);
        }
    }

    private int requiredToken(String token) {
        Integer id = vocabulary.get(token);
        if (id == null) {
            throw new IllegalArgumentException(
                    "vocabulary is missing required token " + token);
        }
        return id;
    }

    public record Encoding(
            long[] inputIds,
            long[] attentionMask,
            long[] tokenTypeIds) {

        public Encoding {
            inputIds = inputIds.clone();
            attentionMask = attentionMask.clone();
            tokenTypeIds = tokenTypeIds.clone();
        }

        @Override
        public long[] inputIds() {
            return inputIds.clone();
        }

        @Override
        public long[] attentionMask() {
            return attentionMask.clone();
        }

        @Override
        public long[] tokenTypeIds() {
            return tokenTypeIds.clone();
        }
    }
}
