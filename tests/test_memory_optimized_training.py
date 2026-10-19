"""CPU regressions for the lazy face trainer; no FFHQ download or full features.

Run from the repository root with ``python -B -m unittest discover -s tests -v``.
The fake generator exercises the real hook/interpolation path at 8x8 pixels.
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "face_keypoints" / "train_modular_memory_optimized.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = load_module("memory_optimized_training_under_test", SCRIPT)
FACE_LABELS = [
    "chin", "left_eye_inner", "left_eye_outer", "left_eye_pupil",
    "left_eyebrow_inner", "left_eyebrow_outer", "lower_lip", "mouth_left",
    "mouth_right", "nose_left", "nose_right", "nose_tip", "right_eye_inner",
    "right_eye_outer", "right_eyebrow_outer", "right_eyebrow_inner",
    "upper_lip", "right_eye_pupil",
]
FACE_STEMS = ["0", "2", "3", "4", "7", "12", "68", "70", "76", "135",
              "160", "250", "251", "344", "439", "543"]


class ConstantFeatures(nn.Module):
    def __init__(self, channels, resolution, first_value):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.first_value = first_value

    def forward(self, inputs):
        values = torch.arange(self.first_value, self.first_value + self.channels,
                              dtype=inputs.dtype, device=inputs.device)
        return values[None, :, None, None].expand(
            inputs.shape[0], self.channels, self.resolution, self.resolution)


class FakeSynthesis(nn.Module):
    block_resolutions = [4, 8]
    num_ws = 3

    def __init__(self):
        super().__init__()
        self.b4 = nn.Module()
        self.b4.conv1 = ConstantFeatures(2, 4, 1)
        self.b8 = nn.Module()
        self.b8.conv0 = ConstantFeatures(3, 8, 3)
        self.b8.conv1 = ConstantFeatures(1, 8, 6)
        self.fail = False

    def forward(self, ws, noise_mode="const"):
        if tuple(ws.shape[1:]) != (3, 2) or noise_mode != "const":
            raise AssertionError("Unexpected W+ shape or noise mode")
        features = self.b4.conv1(ws)
        features = self.b8.conv0(features)
        features = self.b8.conv1(features)
        if self.fail:
            raise RuntimeError("intentional synthesis failure")
        return features.expand(-1, 3, -1, -1)


class FakeGenerator(nn.Module):
    num_ws = 3
    w_dim = 2

    def __init__(self):
        super().__init__()
        self.synthesis = FakeSynthesis()


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 18, 1)

    def forward(self, features):
        return self.conv(features).sigmoid()


class TrainerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="face_training_test_")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(training, "device", "cpu").start()
        self.output = io.StringIO()
        capture = contextlib.redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def processor(self, **overrides):
        arguments = {
            "annotation_dir": str(self.base / "annotations"),
            "projected_dir": str(self.base / "projected"),
            "generator_path": str(self.base / "ffhq.pkl"),
            "stylegan_path": str(self.base / "stylegan2-ada-pytorch"),
        }
        arguments.update(overrides)
        return training.LazyDataProcessor(**arguments)

    def sample(self, stem="0", suffix=".png", annotation=None, projected=True):
        annotation_dir = self.base / "annotations"
        json_dir = annotation_dir / "keypoints_annotation"
        json_dir.mkdir(parents=True, exist_ok=True)
        # Images supply names only; training extracts features from projected W+.
        (annotation_dir / (stem + suffix)).write_bytes(b"")
        if annotation is None:
            annotation = {
                "imageWidth": 1024, "imageHeight": 1024,
                "shapes": [{"label": label, "shape_type": "point",
                            "points": [[128 + index, 256 + index]]}
                           for index, label in enumerate(FACE_LABELS)],
            }
        (json_dir / (stem + ".json")).write_text(
            json.dumps(annotation), encoding="utf-8")
        if projected:
            directory = self.base / "projected" / ("seed" + stem)
            directory.mkdir(parents=True, exist_ok=True)
            np.savez(directory / "projected_w.npz", w=np.zeros((1, 3, 2)))
        return annotation

    def dataset(self, array=None, **npz_contents):
        path = self.base / "latent.npz"
        if not npz_contents:
            if array is None:
                array = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
            npz_contents = {"w": array}
        np.savez(path, **npz_contents)
        keypoints = np.tile([[0.5, 0.25]], (18, 1)).astype(np.float32)
        return training.LazyKeypointDataset(
            [str(path)], [keypoints], FakeGenerator(), self.processor())

    def trainer(self, exp_dir="experiment", resume=None, epochs=2):
        exp_path = self.base / exp_dir
        exp_path.mkdir(parents=True, exist_ok=True)
        trainer = training.Trainer({
            "exp_dir": str(exp_path), "epochs": epochs, "batch_size": 2,
            "lr": 0.001, "patience": 3, "resume": resume,
        })
        trainer.model = TinyPredictor()
        trainer.setup_optimizer()
        return trainer


class FaceDataTests(TrainerTestCase):
    def test_repository_faces_match_legacy_point_order_and_normalization(self):
        self.assertEqual(training.ORIGIN_PIC_SIZE, 1024.0)
        self.assertEqual(training.KEYPOINT_ORDER, FACE_LABELS)
        annotation_dir = ROOT / "face_keypoint_annotation"
        self.assertEqual(len(list(annotation_dir.glob("*.png"))), 16)
        self.assertEqual(len(list((annotation_dir / "keypoints_annotation").glob("*.json"))), 16)
        processor = self.processor(annotation_dir=str(annotation_dir))
        actual = np.stack(processor.load_keypoints_list(FACE_STEMS))
        expected = np.load(ROOT / "face_keypoints" / "keypoints.npy")
        self.assertEqual(actual.shape, (16, 18, 2))
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        for stem in FACE_STEMS:
            annotation = json.loads((annotation_dir / "keypoints_annotation" / (stem + ".json")).read_text(encoding="utf-8"))
            self.assertEqual((annotation["imageWidth"], annotation["imageHeight"]), (1024, 1024))
            self.assertEqual({shape["label"] for shape in annotation["shapes"]}, set(FACE_LABELS))

    def test_natural_sort_and_supported_image_extensions_preserve_pairing(self):
        for stem, suffix in [("12", ".png"), ("2", ".JPG"), ("0", ".jpeg")]:
            self.sample(stem, suffix)
        paths, names = self.processor().get_w_paths_and_filenames()
        self.assertEqual(names, ["0", "2", "12"])
        self.assertEqual([Path(path).parent.name for path in paths], ["seed0", "seed2", "seed12"])

    def test_empty_directory_and_missing_pairs_fail_early(self):
        (self.base / "annotations").mkdir()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.processor().get_w_paths_and_filenames()
        self.sample(projected=False)
        with self.assertRaises(FileNotFoundError):
            self.processor().get_w_paths_and_filenames()
        self.sample()
        (self.base / "annotations" / "keypoints_annotation" / "0.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.processor().get_w_paths_and_filenames()

    def test_duplicate_image_stems_are_rejected(self):
        self.sample("0", ".png")
        self.sample("0", ".jpg")
        with self.assertRaises(ValueError):
            self.processor().get_w_paths_and_filenames()

    def test_json_shape_order_does_not_change_keypoint_order(self):
        annotation = self.sample()
        expected = np.array([[128 + index, 256 + index] for index in range(18)]) / 1024
        annotation["shapes"].reverse()
        self.sample(annotation=annotation)
        np.testing.assert_allclose(self.processor().load_keypoints_list(["0"])[0], expected)

    def test_missing_and_duplicate_labels_are_rejected(self):
        annotation = self.sample()
        removed = annotation["shapes"].pop()
        self.sample(annotation=annotation)
        with self.assertRaises(ValueError):
            self.processor().load_keypoints_list(["0"])
        annotation["shapes"].extend([removed, removed])
        self.sample(annotation=annotation)
        with self.assertRaises(ValueError):
            self.processor().load_keypoints_list(["0"])

    def test_non_face_dimensions_and_non_point_shapes_are_rejected(self):
        for dimension in ("imageWidth", "imageHeight"):
            with self.subTest(dimension=dimension):
                annotation = self.sample()
                annotation[dimension] = 512
                self.sample(annotation=annotation)
                with self.assertRaises(ValueError):
                    self.processor().load_keypoints_list(["0"])
        annotation = self.sample()
        annotation["shapes"][0]["shape_type"] = "circle"
        self.sample(annotation=annotation)
        with self.assertRaises(ValueError):
            self.processor().load_keypoints_list(["0"])

    def test_cli_default_paths_are_relative_to_repository_not_current_directory(self):
        with mock.patch.object(training, "Trainer") as trainer_class, mock.patch("sys.argv", [str(SCRIPT)]):
            training.main()
        configuration = trainer_class.call_args.args[0]
        expected = {
            "annotation_dir": ROOT / "face_keypoint_annotation",
            "projected_dir": ROOT / "face_keypoints" / "projected",
            "generator_path": ROOT / "ffhq.pkl",
            "stylegan_path": ROOT / "stylegan2-ada-pytorch",
        }
        defaults = training.LazyDataProcessor()
        for name, path in expected.items():
            self.assertEqual(Path(configuration[name]).resolve(), path.resolve())
            self.assertEqual(Path(getattr(defaults, name)).resolve(), path.resolve())
        trainer_class.return_value.run.assert_called_once_with()


class LazyFeatureTests(TrainerTestCase):
    def test_latent_with_and_without_batch_dimension_gives_same_features(self):
        with mock.patch.object(training, "FEATURE_MAP_SIZE", 8):
            first, first_heatmaps = self.dataset(np.arange(6, dtype=np.float32).reshape(1, 3, 2))[0]
            second, second_heatmaps = self.dataset(np.arange(6, dtype=np.float32).reshape(3, 2))[0]
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first_heatmaps, second_heatmaps)
        self.assertEqual(tuple(first.shape), (6, 8, 8))
        self.assertEqual(tuple(first_heatmaps.shape), (18, 8, 8))
        self.assertEqual(first.dtype, torch.float32)
        torch.testing.assert_close(first[:, 0, 0], torch.arange(1, 7, dtype=torch.float32))
        torch.testing.assert_close(first, first[:, :1, :1].expand_as(first))
        points = training.extract_keypoints_from_heatmaps(first_heatmaps[None])
        np.testing.assert_allclose(points[0], np.tile([[0.5, 0.25]], (18, 1)))
        self.assertEqual(float(first_heatmaps.max()), 1.0)

    def test_invalid_latent_shapes_and_missing_w_are_rejected(self):
        for shape in [(2, 3, 2), (3, 3), (6,), (1, 2, 3, 2)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.dataset(np.zeros(shape, dtype=np.float32))
        with self.assertRaises((KeyError, ValueError)):
            self.dataset(other=np.zeros((1, 3, 2)))

    def test_hooks_are_removed_after_successful_feature_extraction(self):
        dataset = self.dataset()
        with mock.patch.object(training, "FEATURE_MAP_SIZE", 8):
            dataset[0]
            dataset[0]
        for module in dataset.G.modules():
            self.assertFalse(module._forward_hooks)

    def test_hooks_are_removed_when_synthesis_raises(self):
        dataset = self.dataset()
        dataset.G.synthesis.fail = True
        with mock.patch.object(training, "FEATURE_MAP_SIZE", 8):
            with self.assertRaisesRegex(RuntimeError, "intentional synthesis failure"):
                dataset[0]
        for module in dataset.G.modules():
            self.assertFalse(module._forward_hooks)


class ModelAndCheckpointTests(TrainerTestCase):
    def test_complete_run_saves_aligned_outputs_and_releases_generator(self):
        self.sample("12")
        self.sample("2")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        exp_dir = self.base / "complete_run"
        trainer = training.Trainer({
            "exp_dir": str(exp_dir), "epochs": 1, "batch_size": 2,
            "lr": 0.001, "patience": 3, "resume": None,
            "annotation_dir": str(self.base / "annotations"),
            "projected_dir": str(self.base / "projected"),
            "generator_path": str(self.base / "ffhq.pkl"),
            "stylegan_path": str(self.base / "stylegan2-ada-pytorch"),
        })

        def load_fake_generator(processor):
            processor.G = FakeGenerator().eval().requires_grad_(False)
            return processor.G

        with mock.patch.object(training, "FEATURE_MAP_SIZE", 8):
            with mock.patch.object(training.LazyDataProcessor, "load_generator", load_fake_generator):
                trainer.run()
        for name in ("best_model.pth", "loss_curve.png", "predicted_keypoints.npy",
                     "evaluation_results.png", "filenames.json", "training_config.json"):
            self.assertTrue((exp_dir / name).is_file(), name)
        self.assertEqual(json.loads((exp_dir / "filenames.json").read_text(encoding="utf-8")), ["2", "12"])
        config = json.loads((exp_dir / "training_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["keypoint_order"], FACE_LABELS)
        self.assertEqual(config["in_channels"], 6)
        self.assertEqual(config["num_samples"], 2)
        self.assertEqual(config["origin_pic_size"], 1024)
        predicted = np.load(exp_dir / "predicted_keypoints.npy")
        self.assertEqual(predicted.shape, (2, 18, 2))
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue(((predicted >= 0) & (predicted < 1)).all())
        self.assertIsNone(trainer.dataset.G)
        self.assertIsNone(trainer.processor.G)

    def test_default_predictor_loads_existing_face_checkpoint_and_old_inference(self):
        old_inference = load_module("legacy_inference_under_test", ROOT / "face_keypoints" / "inference.py")
        checkpoint = torch.load(ROOT / "face_keypoints" / "exp_keypoint_epoch93" / "best_model.pth",
                                map_location="cpu", weights_only=True)
        model = training.HeatmapPredictor(verbose=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        legacy = old_inference.HeatmapPredictor()
        legacy.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        legacy.eval()
        features = torch.zeros(1, 5568, 8, 8)
        with torch.no_grad():
            expected = legacy(features)
            actual = model(features)
        self.assertEqual(tuple(actual.shape), (1, 18, 8, 8))
        torch.testing.assert_close(actual, expected)

    def test_cpu_training_optimizer_scheduler_and_amp_checkpoint_round_trip(self):
        trainer = self.trainer()
        features = torch.ones(2, 1, 4, 4)
        heatmaps = torch.zeros(2, 18, 4, 4)
        before = trainer.model.conv.weight.detach().clone()
        loss = trainer.train_epoch(DataLoader(TensorDataset(features, heatmaps), batch_size=2), 0)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, trainer.model.conv.weight))
        self.assertFalse(trainer.scaler.is_enabled())
        trainer.scheduler.step(loss)
        trainer.best_train_loss = loss
        trainer.train_losses = [loss]
        trainer.save_checkpoint(0, loss)
        checkpoint_path = Path(trainer.exp_dir) / "best_model.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.assertIn("scaler_state_dict", checkpoint)
        resumed = self.trainer("resumed", str(checkpoint_path))
        resumed.load_checkpoint()
        self.assertEqual(resumed.start_epoch, 1)
        self.assertEqual(resumed.best_train_loss, loss)
        self.assertEqual(resumed.train_losses, [loss])
        self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
        self.assertEqual(resumed.scaler.state_dict(), trainer.scaler.state_dict())
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[key])
        self.assertEqual(resumed.optimizer.param_groups[0]["lr"], trainer.optimizer.param_groups[0]["lr"])
        self.assertEqual(len(resumed.optimizer.state), len(trainer.optimizer.state))

    def test_legacy_resume_in_new_experiment_can_evaluate_without_new_best(self):
        legacy_model = TinyPredictor()
        for parameter in legacy_model.parameters():
            nn.init.constant_(parameter, 0.0)
        source = self.base / "legacy_best.pth"
        torch.save({"epoch": 0, "model_state_dict": legacy_model.state_dict(), "train_loss": 1e-8}, source)
        trainer = self.trainer("different_experiment", str(source), epochs=2)
        # A pre-existing unrelated file must never override the requested resume.
        torch.save({"epoch": 99, "model_state_dict": trainer.model.state_dict(), "train_loss": 99.0},
                   Path(trainer.exp_dir) / "best_model.pth")
        trainer.dataset = TensorDataset(torch.ones(2, 1, 4, 4), torch.zeros(2, 18, 4, 4))
        trainer.keypoints_list = [np.zeros((18, 2)), np.zeros((18, 2))]
        trainer.processor = mock.Mock()
        trainer.load_checkpoint()
        self.assertEqual(trainer.start_epoch, 1)
        self.assertEqual(trainer.best_train_loss, 1e-8)

        def non_improving_epoch(*_args):
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.fill_(7.0)
            return 1.0

        with mock.patch.object(trainer, "train_epoch", side_effect=non_improving_epoch):
            trainer.train()
        with mock.patch.object(training, "visualize_results"):
            trainer.evaluate()
        for key, value in legacy_model.state_dict().items():
            torch.testing.assert_close(trainer.model.state_dict()[key], value)
        prediction = np.load(Path(trainer.exp_dir) / "predicted_keypoints.npy")
        self.assertEqual(prediction.shape, (2, 18, 2))
        trainer.processor.unload_generator.assert_called_once_with()

    def test_missing_resume_is_an_error(self):
        trainer = self.trainer(resume=str(self.base / "missing_checkpoint.pth"))
        with self.assertRaises(FileNotFoundError):
            trainer.load_checkpoint()


if __name__ == "__main__":
    unittest.main()
