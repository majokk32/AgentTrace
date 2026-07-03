package com.yikeyang.agenttrace.importer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.yikeyang.agenttrace.model.Trajectory;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class AguvisRowApiImporterTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void importsTextOnlyParquetRowFile() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode row = mapper.createObjectNode();
        row.putArray("images").addNull();
        row.put("messages", """
                [{"role":"user","content":[
                  {"type":"image","index":0},
                  {"type":"text","text":"Open Wi-Fi settings"}
                ]}]
                """);
        row.put("metadata", """
                {"platform":"mobile","others":{
                  "id":"aguvis_android_control_wifi_1",
                  "source":"xlangai/aguvis-stage2",
                  "source_id":"wifi_1"
                }}
                """);
        ObjectNode wrapper = mapper.createObjectNode();
        wrapper.put("row_idx", 0);
        wrapper.set("row", row);
        Path rowsFile = temporaryDirectory.resolve("rows.json");
        mapper.writeValue(rowsFile.toFile(), mapper.createArrayNode().add(wrapper));
        Path output = temporaryDirectory.resolve("trajectories.json");

        AguvisRowApiImporter.ImportSummary summary =
                new AguvisRowApiImporter(mapper).importRowsFile(
                        rowsFile, 1, 32, output);
        Trajectory[] trajectories =
                mapper.readValue(output.toFile(), Trajectory[].class);

        assertEquals(1, summary.trajectoryCount());
        assertEquals(32, summary.embeddingDimension());
        assertEquals("aguvis_android_control_wifi_1", trajectories[0].id());
        assertEquals(1, trajectories[0].imageCount());
        assertEquals(32, trajectories[0].embedding().length);
    }
}
