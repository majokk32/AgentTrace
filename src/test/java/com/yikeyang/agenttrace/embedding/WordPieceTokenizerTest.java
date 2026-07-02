package com.yikeyang.agenttrace.embedding;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WordPieceTokenizerTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void lowercasesStripsAccentsAndSplitsWordPieces() throws Exception {
        Path vocabulary = temporaryDirectory.resolve("vocab.txt");
        Files.writeString(vocabulary, String.join("\n",
                "[PAD]",
                "[unused]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "set",
                "alarm",
                "play",
                "##ing",
                "!"));

        WordPieceTokenizer tokenizer = new WordPieceTokenizer(vocabulary, 16);
        WordPieceTokenizer.Encoding encoding =
                tokenizer.encode("Sét alarm playing!");

        assertArrayEquals(
                new long[] {3, 5, 6, 7, 8, 9, 4},
                encoding.inputIds());
        assertArrayEquals(
                new long[] {1, 1, 1, 1, 1, 1, 1},
                encoding.attentionMask());
        assertArrayEquals(new long[7], encoding.tokenTypeIds());
    }
}
