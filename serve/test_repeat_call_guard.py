import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import (REPEAT_NOTE_AFTER, REPEAT_RESULT_SHOWN, REPEAT_STOP_AFTER, ByteTokenizer, MockEngine, Service,
                          repeat_call_guard_setting, repeat_call_verdict, repeat_note, repeat_stop_text, serve,
                          tool_rounds_tail, wants_json_reply, with_repeat_note)

ROOT = Path(__file__).resolve().parents[1]
USER = {"role": "user", "content": "fix the lines"}
NO_CHANGE = "No changes made to a.html. The replacement produced identical content."
TOOLS = [{"name": "edit", "description": "Edit a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
         {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]


def call(name="edit", **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments or {"path": "a.html"}}}]}


def result(text=NO_CHANGE):
    return {"role": "tool", "content": text}


def rounds(count, name="edit", text=NO_CHANGE):
    return [USER] + [m for _ in range(count) for m in (call(name), result(text))]


class RecordingEngine(MockEngine):
    def __init__(self, tokenizer, script, **kwargs):
        super().__init__(tokenizer, script, **kwargs)
        self.prompts, self.samplings = [], []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.samplings.append(dict(sampling or {}))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Setting(unittest.TestCase):
    def test_off_on_and_custom(self):
        self.assertIsNone(repeat_call_guard_setting(False))
        self.assertIsNone(repeat_call_guard_setting(None))
        self.assertEqual(repeat_call_guard_setting(True), (REPEAT_NOTE_AFTER, REPEAT_STOP_AFTER))
        self.assertEqual(repeat_call_guard_setting({"note_after": 3, "stop_after": 5}), (3, 5))
        self.assertEqual(repeat_call_guard_setting({"stop_after": 6}), (REPEAT_NOTE_AFTER, 6))

    def test_a_late_note_moves_the_default_stop_with_it(self):
        self.assertEqual(repeat_call_guard_setting({"note_after": 4}), (4, 4))

    def test_bad_values_are_refused(self):
        for value in (1, "on", {}, {"note_after": 1}, {"stop_after": 1}, {"note_after": 4, "stop_after": 3},
                      {"note_after": True}, {"note_after": 2.0}, {"after": 2}, [2, 3]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                repeat_call_guard_setting(value)


class Tail(unittest.TestCase):
    def test_the_rounds_after_the_last_user_message(self):
        self.assertEqual(len(tool_rounds_tail(rounds(3))), 3)
        self.assertEqual(len(tool_rounds_tail(rounds(2) + rounds(1))), 1)

    def test_a_conversation_that_does_not_end_on_a_tool_result_has_none(self):
        self.assertEqual(tool_rounds_tail(rounds(2) + [USER]), [])
        self.assertEqual(tool_rounds_tail(rounds(2) + [{"role": "assistant", "content": "done"}]), [])
        self.assertEqual(tool_rounds_tail([]), [])
        self.assertEqual(tool_rounds_tail([result()]), [])

    def test_an_answer_without_a_call_ends_the_tail(self):
        messages = rounds(2) + [{"role": "assistant", "content": "let me look"}, result()]
        self.assertEqual(tool_rounds_tail(messages), [])

    def test_a_round_keeps_all_results_of_its_parallel_calls(self):
        both = {"role": "assistant", "content": "", "tool_calls": call("read")["tool_calls"] + call("edit")["tool_calls"]}
        tail = tool_rounds_tail([USER, both, result("one"), result("two")])
        self.assertEqual([len(results) for _, results in tail], [2])


class Verdict(unittest.TestCase):
    def found(self, messages, note=REPEAT_NOTE_AFTER, stop=REPEAT_STOP_AFTER):
        return repeat_call_verdict(messages, note, stop)

    def test_one_round_is_not_a_repeat(self):
        self.assertIsNone(self.found(rounds(1)))
        self.assertIsNone(self.found([USER]))

    def test_the_same_round_twice_gets_a_note_and_three_times_stops(self):
        twice, thrice = self.found(rounds(2)), self.found(rounds(3))
        self.assertEqual((twice.period, twice.times, twice.stop, twice.names), (1, 2, False, ["edit"]))
        self.assertEqual((thrice.period, thrice.times, thrice.stop), (1, 3, True))
        self.assertEqual(thrice.result, NO_CHANGE)

    def test_other_arguments_or_another_result_are_progress(self):
        other_arguments = [USER, call(path="a.html"), result(), call(path="b.html"), result(), call(path="c.html"), result()]
        other_result = [USER, call(), result("pending"), call(), result("running"), call(), result("done")]
        self.assertIsNone(self.found(other_arguments))
        self.assertIsNone(self.found(other_result))

    def test_only_the_identical_rounds_at_the_end_count(self):
        messages = [USER, call("read"), result("x")] + rounds(2)[1:]
        self.assertEqual(self.found(messages).times, 2)

    def test_a_new_user_message_starts_the_count_again(self):
        self.assertIsNone(self.found(rounds(3) + rounds(1)))

    def test_the_key_order_of_the_arguments_does_not_matter(self):
        first = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "edit", "arguments": {"a": 1, "b": 2}}}]}
        second = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "edit", "arguments": {"b": 2, "a": 1}}}]}
        self.assertEqual(self.found([USER, first, result(), second, result()]).times, 2)

    def test_a_block_of_two_rounds_that_repeats(self):
        pair = [call("edit"), result(), call("read"), result("the file")]
        twice, thrice = self.found([USER] + pair * 2), self.found([USER] + pair * 3)
        self.assertEqual((twice.period, twice.times, twice.stop, twice.names), (2, 2, False, ["edit", "read"]))
        self.assertEqual((thrice.period, thrice.times, thrice.stop), (2, 3, True))

    def test_a_block_of_three_rounds_that_repeats(self):
        block = [call("edit"), result(), call("read"), result("the file"), call("read", path="b.html"), result("other")]
        self.assertEqual(self.found([USER] + block * 3).period, 3)
        self.assertIsNone(self.found([USER] + block + block[:4]))

    def test_identical_rounds_are_one_round_repeated_not_a_longer_block(self):
        found = self.found(rounds(4))
        self.assertEqual((found.period, found.times), (1, 4))

    def test_the_thresholds_are_the_configured_ones(self):
        self.assertIsNone(self.found(rounds(3), note=4, stop=6))
        self.assertFalse(self.found(rounds(5), note=4, stop=6).stop)
        self.assertTrue(self.found(rounds(6), note=4, stop=6).stop)

    def test_a_result_with_parts_is_compared_and_quoted_by_its_text(self):
        parts = [{"type": "text", "text": "same"}, {"type": "image", "source": "x.png"}]
        messages = [USER] + [m for _ in range(2) for m in (call(), {"role": "tool", "content": list(parts)})]
        self.assertEqual(self.found(messages).result, "same")


class Texts(unittest.TestCase):
    def test_the_note_names_the_call_and_the_count(self):
        note = repeat_note(repeat_call_verdict(rounds(2), 2, 3))
        self.assertIn("the same tool call (edit)", note)
        self.assertIn("2 times in a row", note)
        self.assertTrue(note.startswith("[strata] Note:"))

    def test_a_block_is_named_as_a_sequence(self):
        pair = [call("edit"), result(), call("read"), result("the file")]
        self.assertIn("the same sequence of tool calls (edit, read)", repeat_note(repeat_call_verdict([USER] + pair * 2, 2, 3)))

    def test_the_stop_text_quotes_a_long_result_cut_short(self):
        text = repeat_stop_text(repeat_call_verdict(rounds(3, text="no  change\n" * 100), 2, 3))
        self.assertIn("3 times in a row", text)
        self.assertIn("no change no change", text)
        self.assertLess(len(text), REPEAT_RESULT_SHOWN + 400)

    def test_the_note_goes_into_a_copy_of_the_last_result(self):
        messages = rounds(2)
        noted = with_repeat_note(messages, "NOTE")
        self.assertEqual(noted[-1]["content"], NO_CHANGE + "\n\nNOTE")
        self.assertEqual(messages[-1]["content"], NO_CHANGE)
        self.assertIs(noted[0], messages[0])

    def test_the_note_is_one_more_text_part_of_a_result_with_parts(self):
        parts = [{"type": "text", "text": "same"}, {"type": "image", "source": "x.png"}]
        messages = [USER, call(), {"role": "tool", "content": parts}]
        noted = with_repeat_note(messages, "NOTE")
        self.assertEqual(noted[-1]["content"][-1], {"type": "text", "text": "\n\nNOTE"})
        self.assertEqual(len(parts), 2)


class InTheService(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = RecordingEngine(self.tok, "</think>\n\nanswer", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.repeat_call_guard = (REPEAT_NOTE_AFTER, REPEAT_STOP_AFTER)

    def ask(self, messages, sampling=None, limit=1024):
        ids, thinking, max_new = self.svc.prepare(messages, TOOLS, {}, limit)
        events = list(self.svc.run(ids, thinking, TOOLS, max_new, sampling if sampling is not None else {"temperature": 0},
                                   threading.Event()))
        content = "".join(e.text or "" for kind, e in events if kind == "event" and e.kind == "content")
        return ids, content, events[-1][1]

    def test_one_round_is_left_alone(self):
        ids, content, done = self.ask(rounds(1))
        self.assertNotIn("[strata]", self.tok.decode(ids))
        self.assertEqual(content, "answer")
        self.assertEqual(self.engine.samplings, [{"temperature": 0}])

    def test_the_second_identical_round_adds_the_note_and_samples_the_reply(self):
        ids, content, done = self.ask(rounds(2))
        prompt = self.tok.decode(ids)
        self.assertEqual(prompt.count("[strata] Note: the same tool call (edit) has now been made 2 times"), 1)
        self.assertLess(prompt.rindex(NO_CHANGE), prompt.index("[strata] Note:"))
        self.assertEqual(content, "answer")
        self.assertEqual(done["finish"], "stop")
        self.assertEqual((self.engine.samplings[0]["temperature"], self.engine.samplings[0]["presence_penalty"]), (1.0, 1.5))

    def test_the_note_stage_leaves_the_request_and_its_messages_unchanged(self):
        messages, sampling = rounds(2), {"temperature": 0, "top_p": 0.5}
        self.ask(messages, sampling)
        self.assertEqual(messages, rounds(2))
        self.assertEqual(sampling, {"temperature": 0, "top_p": 0.5})
        self.assertEqual(self.engine.samplings[0]["top_p"], 0.5)

    def test_the_third_identical_round_ends_the_reply_without_the_engine(self):
        ids, content, done = self.ask(rounds(3))
        self.assertEqual(self.engine.prompts, [])
        self.assertTrue(content.startswith("[strata] The reply was stopped to end a loop"))
        self.assertIn("(edit) 3 times in a row", content)
        self.assertIn("No changes made to a.html.", content)
        self.assertEqual(done["finish"], "stop")
        self.assertEqual(done["completion_tokens"], len(self.tok.encode(content)))
        self.assertNotIn("[strata]", self.tok.decode(ids))

    def test_the_notice_keeps_to_the_max_tokens_of_the_request(self):
        whole = repeat_stop_text(repeat_call_verdict(rounds(3), 2, 3))
        ids, content, done = self.ask(rounds(3), limit=40)
        self.assertEqual((content, done["finish"], done["completion_tokens"]), (whole[:40], "length", 40))
        self.assertEqual(self.engine.prompts, [])

    def test_a_character_cut_by_the_limit_is_left_out(self):
        whole = repeat_stop_text(repeat_call_verdict(rounds(3, text="Нет изменений"), 2, 3))
        cut = len(whole[:whole.index("Нет") + 2].encode()) - 1          # in the middle of the second Cyrillic letter
        ids, content, done = self.ask(rounds(3, text="Нет изменений"), limit=cut)
        self.assertEqual(content, whole[:whole.index("Нет") + 1])
        self.assertEqual(done["completion_tokens"], cut)

    def test_the_notice_stops_at_a_stop_string_of_the_request(self):
        ids, content, done = self.ask(rounds(3), {"stop": ["times in a row"]})
        self.assertTrue(content.endswith("(edit) 3 "))
        self.assertEqual((done["finish"], done["stop_sequence"]), ("stop", "times in a row"))

    def test_a_stopped_request_is_counted_with_no_engine_work(self):
        before = self.svc.totals["requests"]
        ids, content, done = self.ask(rounds(3))
        row = self.svc.history[-1]
        self.assertEqual(self.svc.totals["requests"], before + 1)
        self.assertEqual((row["finish"], row["repeated_calls"], row["engine_generated"], row["prompt_total"]),
                         ("stop", 3, 0, len(ids)))
        self.assertEqual(row["output_tokens"], done["completion_tokens"])

    def test_a_request_without_tools_cannot_repeat_a_call(self):
        ids, thinking, max_new = self.svc.prepare(rounds(3), None, {}, 1024)
        events = list(self.svc.run(ids, thinking, None, max_new, {}, threading.Event()))
        self.assertNotIn("[strata]", self.tok.decode(ids))
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(events[-1][1]["finish"], "stop")

    def test_a_request_for_json_is_refused_instead_of_answered_with_the_notice(self):
        for req in ({"text": {"format": {"type": "json_object"}}}, {"response_format": {"type": "json_schema", "json_schema": {}}}):
            with self.subTest(req=req), self.assertRaises(ValueError) as raised:
                self.svc.prepare(rounds(3), TOOLS, {}, 1024, req=req)
            self.assertIn("[strata] The reply was stopped to end a loop", str(raised.exception))
            self.assertIsNone(self.svc.repeat_local.found)
        self.svc.prepare(rounds(3), TOOLS, {}, 1024, req={"response_format": {"type": "text"}})
        self.assertTrue(self.svc.repeat_local.found.stop)

    def test_a_request_for_json_still_gets_the_note(self):
        ids, _, _ = self.svc.prepare(rounds(2), TOOLS, {}, 1024, req={"text": {"format": {"type": "json_object"}}})
        self.assertIn("[strata] Note:", self.tok.decode(ids))

    def test_what_counts_as_a_request_for_json(self):
        self.assertTrue(wants_json_reply({"response_format": {"type": "json_object"}}))
        self.assertTrue(wants_json_reply({"text": {"format": {"type": "json_schema"}}}))
        for req in (None, {}, {"response_format": {"type": "text"}}, {"response_format": "json"}, {"text": "x"},
                    {"text": {"format": None}}):
            self.assertFalse(wants_json_reply(req))

    def test_off_by_default(self):
        self.svc.repeat_call_guard = Service(self.engine, self.tok, None).repeat_call_guard
        self.assertIsNone(self.svc.repeat_call_guard)
        ids, content, done = self.ask(rounds(5))
        self.assertNotIn("[strata]", self.tok.decode(ids))
        self.assertEqual(content, "answer")
        self.assertEqual(self.engine.samplings, [{"temperature": 0}])

    def test_a_finding_is_used_by_one_run_only(self):
        ids, content, _ = self.ask(rounds(3))
        self.assertEqual(self.engine.prompts, [])
        events = list(self.svc.run(ids, True, TOOLS, 64, {}, threading.Event()))     # no prepare() before this one
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(events[-1][1]["finish"], "stop")

    def test_the_next_request_starts_clean(self):
        self.svc.prepare(rounds(3), TOOLS, {}, 64)              # a request whose run never started
        ids, content, done = self.ask([USER])
        self.assertEqual(content, "answer")
        self.assertEqual(len(self.engine.prompts), 1)

    def test_a_refused_request_leaves_no_finding(self):
        self.engine.max_context = 64
        with self.assertRaises(ValueError):
            self.svc.prepare(rounds(3), TOOLS, {}, 16)
        self.assertIsNone(self.svc.repeat_local.found)


class OverHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\nanswer", max_context=16384)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.repeat_call_guard = (REPEAT_NOTE_AFTER, REPEAT_STOP_AFTER)
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.engine.prompts.clear()
        self.engine.samplings.clear()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    @staticmethod
    def chat(count, **more):
        messages = [{"role": "user", "content": "fix the lines"}]
        for i in range(count):
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"call_{i}", "type": "function", "function": {"name": "edit", "arguments": json.dumps({"path": "a.html"})}}]})
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": NO_CHANGE})
        return {"model": "m", "messages": messages, "max_tokens": 1024, "tools": [{"type": "function", "function": t} for t in TOOLS], **more}

    def test_chat_completions_end_the_loop_with_a_text_reply(self):
        status, body = self.post("/v1/chat/completions", self.chat(3))
        choice = json.loads(body)["choices"][0]
        self.assertEqual(status, 200)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertTrue(choice["message"]["content"].startswith("[strata] The reply was stopped to end a loop"))
        self.assertFalse(choice["message"].get("tool_calls"))
        self.assertEqual(self.engine.prompts, [])

    def test_a_stream_ends_the_loop_the_same_way(self):
        status, body = self.post("/v1/chat/completions", self.chat(3, stream=True))
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
        finishes = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")]
        self.assertEqual(status, 200)
        self.assertTrue(text.startswith("[strata] The reply was stopped to end a loop"))
        self.assertIn("stop", finishes)
        self.assertNotIn("tool_calls", finishes)
        self.assertEqual(self.engine.prompts, [])

    def test_two_rounds_reach_the_engine_with_the_note(self):
        status, body = self.post("/v1/chat/completions", self.chat(2))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "answer")
        self.assertIn("[strata] Note:", ByteTokenizer().decode(self.engine.prompts[0]))
        self.assertEqual(self.engine.samplings[0]["temperature"], 1.0)

    def test_the_call_ids_of_a_client_do_not_hide_a_repeat(self):
        request = self.chat(3)
        self.assertEqual(len({m["tool_calls"][0]["id"] for m in request["messages"] if m.get("tool_calls")}), 3)
        status, body = self.post("/v1/chat/completions", request)
        self.assertIn("[strata] The reply was stopped", json.loads(body)["choices"][0]["message"]["content"])

    def test_a_small_max_tokens_cuts_the_notice_and_still_ends_the_loop(self):
        status, body = self.post("/v1/chat/completions", self.chat(3, max_tokens=24))
        choice = json.loads(body)["choices"][0]
        self.assertEqual((status, choice["finish_reason"]), (200, "length"))
        self.assertEqual(choice["message"]["content"], "[strata] The reply was s")
        self.assertFalse(choice["message"].get("tool_calls"))
        self.assertEqual(json.loads(body)["usage"]["completion_tokens"], 24)

    def test_responses_with_a_json_format_are_refused_with_the_reason(self):
        items = [{"role": "user", "content": "fix the lines"}]
        for i in range(3):
            items.append({"type": "function_call", "call_id": f"call_{i}", "name": "edit", "arguments": json.dumps({"path": "a.html"})})
            items.append({"type": "function_call_output", "call_id": f"call_{i}", "output": NO_CHANGE})
        tools = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in TOOLS]
        request = {"model": "m", "input": items, "tools": tools, "max_output_tokens": 1024}
        status, body = self.post("/v1/responses", {**request, "text": {"format": {"type": "json_object"}}})
        self.assertEqual(status, 400)
        self.assertIn("The reply was stopped to end a loop", json.loads(body)["error"]["message"])
        status, body = self.post("/v1/responses", request)
        texts = [part.get("text", "") for item in json.loads(body)["output"] for part in item.get("content") or []]
        self.assertEqual(status, 200)
        self.assertTrue("".join(texts).startswith("[strata] The reply was stopped to end a loop"))
        self.assertEqual(self.engine.prompts, [])

    def test_anthropic_messages_end_the_loop_with_a_text_block(self):
        messages = [{"role": "user", "content": "fix the lines"}]
        for i in range(3):
            messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"toolu_{i}", "name": "edit", "input": {"path": "a.html"}}]})
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"toolu_{i}", "content": NO_CHANGE}]})
        tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in TOOLS]
        status, body = self.post("/v1/messages", {"model": "m", "messages": messages, "max_tokens": 1024, "tools": tools})
        reply = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(reply["stop_reason"], "end_turn")
        self.assertEqual([b["type"] for b in reply["content"]], ["text"])
        self.assertTrue(reply["content"][0]["text"].startswith("[strata] The reply was stopped to end a loop"))
        self.assertEqual(self.engine.prompts, [])


if __name__ == "__main__":
    unittest.main()
