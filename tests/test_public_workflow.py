import tempfile
import unittest
from pathlib import Path

from src.pipeline import (
    build_conditioned_intervals,
    load_processed,
    source_quality,
)
from src.synthetic_data import generate_synthetic_data


class TestPublicWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.path = (
            Path(cls._temporary_directory.name) / "synthetic_asset_events.csv"
        )
        generate_synthetic_data(cls.path)
        cls.frame = load_processed(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._temporary_directory.cleanup()

    def test_synthetic_identifiers_are_non_operational(self):
        self.assertTrue(self.frame["MOTOR"].str.fullmatch(r"SYN-\d{4}").all())
        self.assertTrue(self.frame["Nombre"].str.startswith("Synthetic motor").all())

    def test_source_invariants(self):
        quality = source_quality(self.frame)
        self.assertEqual(quality["registros"], 360)
        self.assertEqual(quality["motores"], 60)
        self.assertEqual(quality["fallas_etiquetadas"], 180)
        self.assertEqual(quality["mantenimientos_etiquetados"], 180)
        self.assertEqual(quality["intervalos_negativos_en_orden_archivo"], 0)

    def test_conditioned_target(self):
        intervals = build_conditioned_intervals(self.frame)
        self.assertEqual(len(intervals), 120)
        self.assertEqual(intervals["MOTOR_NORMALIZADO"].nunique(), 60)
        self.assertTrue((intervals["objetivo_h"] > 0).all())


if __name__ == "__main__":
    unittest.main()
