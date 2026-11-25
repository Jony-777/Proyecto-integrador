"""Pruebas del flujo de preparación y segmentación."""

import unittest

import pandas as pd

from src.analysis import RAW_DATA_PATH, prepare_data, run_analysis


class PrepareDataTests(unittest.TestCase):
    def test_extracts_capacity_from_text_with_embedded_title(self) -> None:
        source = pd.read_csv(RAW_DATA_PATH)
        prepared, audit = prepare_data(source)
        recovered = prepared.loc[
            prepared["energy_capacity"].str.contains("Toyota Passo", na=False),
            "engine_cc",
        ]
        self.assertEqual(recovered.tolist(), [1000.0])
        self.assertEqual(audit["capacity_values_recovered_from_embedded_text"], 1)

    def test_rejects_unparseable_required_numeric_values(self) -> None:
        source = pd.read_csv(RAW_DATA_PATH).head(2)
        source.loc[source.index[0], "price"] = "precio desconocido"
        with self.assertRaisesRegex(ValueError, "price"):
            prepare_data(source)


class SegmentationTests(unittest.TestCase):
    def test_analysis_evaluates_five_k_values_and_labels_every_record(self) -> None:
        data, results = run_analysis(save_outputs=False)
        self.assertEqual([row["k"] for row in results["evaluacion_k"]], [2, 3, 4, 5, 6])
        self.assertEqual(results["k_seleccionado"], results["metricas_finales"]["k"])
        self.assertEqual(len(data), results["dataset"]["registros"])
        self.assertTrue(data["cluster"].notna().all())
        self.assertEqual(data["cluster"].nunique(), results["k_seleccionado"])
        self.assertTrue(data["cluster_name"].notna().all())
        self.assertEqual(len(results["perfiles_clusters"]), results["k_seleccionado"])


if __name__ == "__main__":
    unittest.main()
