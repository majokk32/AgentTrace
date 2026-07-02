package com.yikeyang.agenttrace.importer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.yikeyang.agenttrace.model.Trajectory;
import org.junit.jupiter.api.Test;

class AguvisRowParserTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void parsesARealisticRowApiRecordWithoutDownloadingImages() throws Exception {
        ObjectNode row = objectMapper.createObjectNode();
        row.putArray("images").addObject();
        row.withArray("images").addObject();
        row.put("messages", """
                [
                  {"role":"user","content":[
                    {"type":"image","index":0},
                    {"type":"text","text":"Set an alarm for 5 PM"}
                  ]},
                  {"role":"assistant",
                   "tool_calls":[{"type":"function","function":{
                     "name":"tap","arguments":{"coordinate":[50,80]}
                   }}],
                   "content":[{"type":"action_description","text":"Open the Clock app."}]},
                  {"role":"assistant",
                   "tool_calls":[{"type":"function","function":{
                     "name":"terminate","arguments":{"status":"success"}
                   }}],
                   "content":[]}
                ]
                """);
        row.put("metadata", """
                {
                  "platform":"mobile",
                  "task_type":"navigation",
                  "others":{
                    "id":"aguvis_android_control_alarm_1",
                    "source":"xlangai/aguvis-stage2",
                    "source_id":"alarm_1"
                  }
                }
                """);
        ObjectNode wrapper = objectMapper.createObjectNode();
        wrapper.put("row_idx", 7);
        wrapper.set("row", row);

        Trajectory trajectory =
                new AguvisRowParser(objectMapper, 256).parse(wrapper);

        assertEquals("aguvis_android_control_alarm_1", trajectory.id());
        assertEquals("Set an alarm for 5 PM", trajectory.instruction());
        assertEquals("android-control", trajectory.app());
        assertEquals("xlangai/aguvis-stage2", trajectory.source());
        assertEquals(2, trajectory.imageCount());
        assertEquals(256, trajectory.embedding().length);
        assertTrue(trajectory.success());
        assertTrue(trajectory.actions().getFirst().startsWith("tap:"));
    }
}
