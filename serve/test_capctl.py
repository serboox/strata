import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import capctl, cpucap
from serve.capctl import CapController, next_cap_mhz
from serve.cpucap import CpuCap
from serve.test_cpucap import isolate_state, make_cpus


class NextCapTests(unittest.TestCase):
    def test_a_hot_lead_lowers_the_cap_at_once(self):
        self.assertEqual(next_cap_mhz(2200, 78.5, 77, 0.0, 1800, 2600, 100), 2100)

    def test_a_lead_far_over_lowers_it_two_steps(self):
        self.assertEqual(next_cap_mhz(2200, 80.5, 77, 0.0, 1800, 2600, 100), 2000)

    def test_a_cool_package_raises_it_only_after_the_wait(self):
        self.assertEqual(next_cap_mhz(2200, 70, 77, 2.0, 1800, 2600, 100), 2200)
        self.assertEqual(next_cap_mhz(2200, 70, 77, 6.0, 1800, 2600, 100), 2300)

    def test_inside_the_band_the_cap_holds(self):
        self.assertEqual(next_cap_mhz(2200, 76.5, 77, 60.0, 1800, 2600, 100), 2200)

    def test_the_cap_stays_in_its_range(self):
        self.assertEqual(next_cap_mhz(1800, 85, 77, 0.0, 1800, 2600, 100), 1800)
        self.assertEqual(next_cap_mhz(2600, 60, 77, 60.0, 1800, 2600, 100), 2600)


class FakeCap:
    def __init__(self, mhz):
        self.max_mhz = mhz
        self.moves = []

    def set_mhz(self, mhz):
        self.max_mhz = mhz
        self.moves.append(mhz)
        return True


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.temp = 70.0
        self.cap = FakeCap(2100)
        self.messages = []
        self.controller = CapController(self.cap, lambda: {"cpu": self.temp}, log=self.messages.append, clock=lambda: self.now)

    def step(self, seconds=1.0):
        self.now += seconds
        return self.controller.tick()

    def test_it_climbs_while_cool_then_backs_off_when_hot(self):
        for _ in range(12):
            self.step()
        self.assertGreater(self.controller.mhz, 2100)
        self.temp = 82.0
        before = self.controller.mhz
        self.step()
        self.assertLess(self.controller.mhz, before)

    def test_a_rising_temperature_counts_as_the_one_it_will_have(self):
        self.controller._changed_at = self.now - 60
        for temp in (72.0, 73.5, 75.0, 76.5):
            self.temp = temp
            self.step()
        self.assertLess(self.controller.mhz, 2100 + 100 * 4)
        self.assertTrue(any(move < 2100 for move in self.cap.moves) or self.cap.moves[-1] <= 2100)

    def test_no_sensor_keeps_the_cap(self):
        controller = CapController(self.cap, lambda: {}, log=self.messages.append, clock=lambda: self.now)
        self.assertEqual(controller.tick(), 2100)
        self.assertEqual(self.cap.moves, [])

    def test_the_start_value_is_inside_the_range(self):
        controller = CapController(FakeCap(3300), lambda: {}, clock=lambda: 0.0)
        self.assertEqual(controller.mhz, capctl.DEFAULT_MAX_MHZ)


class CreateTests(unittest.TestCase):
    def test_absent_or_without_a_cap_is_none(self):
        self.assertIsNone(capctl.create({}, FakeCap(2100), lambda: {}))
        self.assertIsNone(capctl.create({"cpu_adaptive": True}, None, lambda: {}))

    def test_true_gives_the_defaults(self):
        controller = capctl.create({"cpu_adaptive": True}, FakeCap(2100), lambda: {})
        self.assertEqual((controller.target_c, controller.lo, controller.hi, controller.step), (77.0, 1800.0, 2600.0, 100.0))

    def test_bad_values_are_refused(self):
        for block in ({"min_mhz": 3000, "max_mhz": 2000}, {"step_mhz": 0}, {"target_c": "hot"}, "yes"):
            with self.assertRaises(ValueError):
                capctl.create({"cpu_adaptive": block}, FakeCap(2100), lambda: {})


class SetMhzTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        isolate_state(self)
        self.files = make_cpus(self.tmp.name, count=2)
        self.cap = CpuCap(2100, log=lambda message: None, root=self.tmp.name)
        self.cap.start()

    def test_the_new_value_is_written_to_every_cpu(self):
        self.assertTrue(self.cap.set_mhz(2300))
        self.assertEqual([path.read_text() for path in self.files], ["2300000"] * 2)

    def test_stop_still_puts_the_original_back(self):
        self.cap.set_mhz(2300)
        self.cap.stop()
        self.assertEqual([path.read_text() for path in self.files], ["3300000"] * 2)

    def test_a_limit_changed_by_somebody_else_is_left_alone(self):
        self.files[0].write_text("1500000", encoding="ascii")
        self.cap.set_mhz(2300)
        self.assertEqual(self.files[0].read_text(), "1500000")
        self.assertEqual(self.files[1].read_text(), "2300000")

    def test_the_record_for_the_guard_follows_the_move(self):
        self.cap.set_mhz(2300)
        record = cpucap._load_state(self.state)
        self.assertEqual({values[1] for values in record["held"].values()}, {"2300000"})


if __name__ == "__main__":
    unittest.main()
