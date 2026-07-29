import torch

from cabin_speech.fusion import MultimodalFusionNet
from cabin_speech.synthetic import SyntheticCabinDataset


def test_synthetic_dataset_shapes() -> None:
    dataset = SyntheticCabinDataset(size=16, seed=3)
    features, label = dataset[0]
    assert features.shape == (14,)
    assert label.ndim == 0
    assert 0 <= int(label) < 5


def test_fusion_network_output_shape() -> None:
    model = MultimodalFusionNet(input_dim=14, num_classes=5)
    output = model(torch.randn(8, 14))
    assert output.shape == (8, 5)
