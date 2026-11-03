import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strata_launch


def tensor(name: str, type_name: str):
    return types.SimpleNamespace(name=name, type_name=type_name)


def fake_reader(*tensors):
    return types.SimpleNamespace(GGUFFile=lambda path: types.SimpleNamespace(tensors=list(tensors)))


class ExpertDetectionTests(unittest.TestCase):
    def detect(self, *tensors):
        with mock.patch.dict(sys.modules, {"gguf_reader": fake_reader(*tensors)}):
            return strata_launch.uses_iq_experts(Path("model.gguf"))

    def test_iq3_s_gate_and_up_experts_are_found(self):
        self.assertTrue(self.detect(tensor("blk.0.ffn_gate_exps.weight", "IQ3_S"), tensor("blk.0.ffn_down_exps.weight", "IQ4_XS")))

    def test_q4_k_experts_are_not_iq_experts(self):
        self.assertFalse(self.detect(tensor("blk.0.ffn_gate_exps.weight", "Q4_K"), tensor("blk.0.ffn_down_exps.weight", "Q5_K")))

    def test_an_iq_tensor_outside_the_experts_does_not_count(self):
        self.assertFalse(self.detect(tensor("blk.0.attn_q.weight", "IQ4_XS"), tensor("blk.0.ffn_gate_exps.weight", "Q4_K")))


class KernelEnvTests(unittest.TestCase):
    model = Path("model.gguf")

    def test_off_is_set_explicitly_so_an_inherited_value_cannot_turn_the_kernels_on(self):
        self.assertEqual(strata_launch.iq_kernel_env("off", self.model), {"STRATA_IQ_VBMI": "0"})

    def test_on_does_not_look_at_the_model(self):
        with mock.patch.object(strata_launch, "uses_iq_experts", side_effect=AssertionError("must not read the file")):
            self.assertEqual(strata_launch.iq_kernel_env("on", self.model), {"STRATA_IQ_VBMI": "1"})

    def test_auto_follows_the_experts_of_the_model(self):
        with mock.patch.object(strata_launch, "uses_iq_experts", return_value=True):
            self.assertEqual(strata_launch.iq_kernel_env("auto", self.model), {"STRATA_IQ_VBMI": "1"})
        with mock.patch.object(strata_launch, "uses_iq_experts", return_value=False):
            self.assertIsNone(strata_launch.iq_kernel_env("auto", self.model))


class ModelChoiceTests(unittest.TestCase):
    def test_the_message_names_the_state_of_the_kernels(self):
        self.assertEqual([strata_launch.iq_kernel_state(env) for env in (strata_launch.IQ_KERNELS_ENV, strata_launch.IQ_KERNELS_OFF_ENV, None)],
                         ["on", "off", "not needed"])

    def test_a_model_outside_the_model_directory_is_refused_because_the_tokenizer_would_not_fit(self):
        with mock.patch.object(sys, "argv", ["strata_launch", "--model", "/tmp/other-model.gguf", "--dry-run"]), \
                mock.patch.object(strata_launch, "MODEL_FILE", strata_launch.MODEL_FILE), self.assertRaises(SystemExit) as raised:
            strata_launch.main()
        self.assertEqual(raised.exception.code, 2)


class ConfigTests(unittest.TestCase):
    profile = strata_launch.PROFILES["auto"]

    def test_the_config_has_no_env_unless_asked(self):
        self.assertNotIn("env", strata_launch.build_config(dict(self.profile), 8080, "greedy"))

    def test_the_kernel_switch_reaches_the_engine_environment(self):
        config = strata_launch.build_config(dict(self.profile), 8080, "greedy", 80.0, strata_launch.IQ_KERNELS_ENV)
        self.assertEqual(config["env"], {"STRATA_IQ_VBMI": "1"})
        self.assertEqual(config["thermal"], {"max_temp_c": 80.0})

    def test_the_model_given_on_the_command_line_is_served(self):
        args = strata_launch.build_parser().parse_args(["--model", "/tmp/other.gguf"])
        self.assertEqual((args.model, args.iq_kernels), (Path("/tmp/other.gguf"), "auto"))
        with mock.patch.object(strata_launch, "MODEL_FILE", Path("/tmp/other.gguf")):
            engine = strata_launch.engine_args(dict(self.profile))
        self.assertEqual(engine[engine.index("--native") + 1], "/tmp/other.gguf")

    def test_without_arguments_the_model_and_the_kernels_are_as_before(self):
        args = strata_launch.build_parser().parse_args([])
        self.assertIsNone(args.model)
        self.assertEqual(args.iq_kernels, "auto")


if __name__ == "__main__":
    unittest.main()
