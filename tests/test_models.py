"""Contract checks use only tiny random encoders and never download weights."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from src.losses import CBFocalLoss, build_loss
from src.models import ATTRIBUTE_CLASSES, build_model
from src.models.encoders import ConvNeXtBaseEncoder, TinyCNNEncoder
from src.models.fusion import MeanFusion


def tiny_config(modalities=("clinical", "dermoscopic"), fusion="mean", **updates):
    config = {"modalities": list(modalities), "clinical_encoder": "tiny_cnn",
              "dermoscopic_encoder": "tiny_cnn", "pretrained_id": "none", "fusion": fusion}
    return dict(config, **updates)


@pytest.mark.parametrize("modalities", [("clinical",), ("dermoscopic",), ("clinical", "dermoscopic")])
def test_model_shapes_independent_encoders_and_backward(modalities):
    model = build_model(tiny_config(modalities))
    images = {name: torch.randn(2, 3, 24, 32) for name in modalities}
    outputs = model(**images)
    assert outputs["diagnosis_logits"].shape == (2, 5)
    assert outputs["attribute_logits"] == {}
    assert (model.clinical_encoder is None) == ("clinical" not in modalities)
    assert (model.dermoscopic_encoder is None) == ("dermoscopic" not in modalities)
    if len(modalities) == 2:
        assert next(model.clinical_encoder.parameters()).data_ptr() != next(model.dermoscopic_encoder.parameters()).data_ptr()
    loss = build_loss({})(outputs, {"diagnosis": torch.tensor([0, 2])})
    loss["total"].backward()
    assert all(parameter.grad is not None for parameter in model.encoder_parameters())


def test_plain_mean_is_exact_pooled_mean_and_accepts_different_spatial_shapes():
    clinical, dermoscopic = torch.randn(2, 7), torch.randn(2, 7)
    fused, info = MeanFusion(7, 7)(clinical_tokens=clinical[:, None].expand(-1, 6, -1),
                                  dermoscopic_tokens=dermoscopic[:, None].expand(-1, 11, -1),
                                  clinical_pooled=clinical, dermoscopic_pooled=dermoscopic)
    torch.testing.assert_close(fused, (clinical + dermoscopic) / 2)
    assert info == {}
    assert list(MeanFusion(7, 7).parameters()) == []


@pytest.mark.parametrize("fusion", ["mean", "concat", "gated"])
def test_fusions_forward_with_unregistered_images(fusion):
    model = build_model(tiny_config(fusion=fusion))
    result = model(clinical=torch.randn(2, 3, 20, 28), dermoscopic=torch.randn(2, 3, 32, 40),
                   modality_mask=torch.ones(2, 2, dtype=torch.bool), return_analysis=True)
    assert result["diagnosis_logits"].shape == (2, 5)
    assert result["analysis"]["clinical_tokens_shape"][1] != result["analysis"]["dermoscopic_tokens_shape"][1]


def test_missing_inputs_and_target_leakage_fail_explicitly():
    model = build_model(tiny_config())
    image = torch.randn(2, 3, 16, 16)
    with pytest.raises(ValueError, match="missing"):
        model(clinical=image)
    with pytest.raises(ValueError, match="required modality"):
        model(clinical=image, dermoscopic=image,
              modality_mask=torch.tensor([[True, False], [True, True]]))
    with pytest.raises(TypeError):
        model(clinical=image, dermoscopic=image, targets={"diagnosis": torch.tensor([0, 1])})
    with pytest.raises(ValueError, match="metadata=None"):
        model(clinical=image, dermoscopic=image, metadata=torch.zeros(2, 2))


def test_single_modality_mask_does_not_require_unused_modality():
    model = build_model(tiny_config(("clinical",)))
    output = model(clinical=torch.randn(2, 3, 16, 16),
                   modality_mask=torch.tensor([[True, False], [True, False]]))
    assert output["diagnosis_logits"].shape == (2, 5)


def test_attributes_have_fixed_categorical_sizes():
    model = build_model(tiny_config(("dermoscopic",), attribute_head=True))
    outputs = model(dermoscopic=torch.randn(2, 3, 16, 16))
    assert list(outputs["attribute_logits"]) == list(ATTRIBUTE_CLASSES)
    assert [value.shape[1] for value in outputs["attribute_logits"].values()] == [3, 2, 3, 3, 3, 3, 2]


def test_all_masked_attributes_return_finite_zero_without_reading_invalid_labels():
    outputs = {"diagnosis_logits": torch.randn(2, 5, requires_grad=True),
               "attribute_logits": {name: torch.full((2, count), float("nan"))
                                    for name, count in ATTRIBUTE_CLASSES.items()}}
    targets = {"diagnosis": torch.tensor([1, 2]),
               "attributes": {name: torch.full((2,), -99, dtype=torch.long) for name in ATTRIBUTE_CLASSES},
               "attribute_masks": {name: torch.zeros(2, dtype=torch.bool) for name in ATTRIBUTE_CLASSES}}
    loss = build_loss({"attribute_lambda": 0.25})(outputs, targets)
    assert loss["attributes"].item() == 0.0
    assert torch.isfinite(loss["total"])
    loss["total"].backward()


def test_attribute_reduction_means_valid_tasks_without_masked_examples():
    first = torch.tensor([[2., 0., -1.], [0., 0., 0.]], requires_grad=True)
    second = torch.tensor([[0., 1.], [2., 0.]], requires_grad=True)
    outputs = {"diagnosis_logits": torch.randn(2, 5), "attribute_logits": {"first": first, "second": second}}
    targets = {"diagnosis": torch.tensor([0, 4]),
               "attributes": {"first": torch.tensor([0, 999]), "second": torch.tensor([1, 0])},
               "attribute_masks": {"first": torch.tensor([True, False]), "second": torch.tensor([True, True])}}
    losses = build_loss({"attribute_lambda": 0.5})(outputs, targets)
    expected = (torch.nn.functional.cross_entropy(first[:1], targets["attributes"]["first"][:1])
                + torch.nn.functional.cross_entropy(second, targets["attributes"]["second"])) / 2
    torch.testing.assert_close(losses["attributes"], expected)
    torch.testing.assert_close(losses["total"], losses["diagnosis"] + 0.5 * expected)


def test_cb_focal_reduces_to_ce_at_gamma_beta_zero_and_has_finite_gradient():
    logits = torch.tensor([[100., -100., 0.], [-90., 80., 1.]], requires_grad=True)
    labels = torch.tensor([1, 2])
    focal = CBFocalLoss([2, 10, 0], beta=0.0, gamma=0.0)(logits, labels)
    torch.testing.assert_close(focal, torch.nn.functional.cross_entropy(logits, labels))
    focal.backward()
    assert torch.isfinite(logits.grad).all()


def test_cb_focal_weights_remain_finite_when_beta_is_near_one():
    criterion = CBFocalLoss([1, 100, 1000], beta=1.0 - 1e-12)
    assert torch.isfinite(criterion.alpha).all()
    torch.testing.assert_close(criterion.alpha.mean(), torch.tensor(1.0))


def test_null_pretraining_is_safe_and_single_alias_has_no_fusion():
    model = build_model(tiny_config(("clinical",), fusion="single", pretrained_id=None))
    assert model.clinical_encoder.pretrained_id == "none"
    assert model.fusion is None


def test_inconsistent_configuration_fails():
    with pytest.raises(ValueError, match="feature_dim"):
        build_model(tiny_config(feature_dim=1024))
    with pytest.raises(ValueError, match="independent"):
        build_model(tiny_config(independent_encoders=False))
    with pytest.raises(ValueError, match="pretrained"):
        TinyCNNEncoder(pretrained_id="ConvNeXt_Base_Weights.IMAGENET1K_V1")
    with pytest.raises(ValueError, match="training partition"):
        build_loss({"diagnosis": "cb_focal", "cb_beta": 0.99, "focal_gamma": 2.0})


def test_convnext_adapter_uses_concrete_enum_and_reports_actual_feature_dim(monkeypatch):
    # Exercise the adapter without constructing a 350 MB backbone or fetching weights.
    concrete_weights = object()
    calls = []

    def fake_backbone(*, weights):
        calls.append(weights)
        backbone = torch.nn.Module()
        backbone.features = torch.nn.Conv2d(3, 13, kernel_size=1)
        backbone.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        backbone.classifier = torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.Linear(13, 1000))
        return backbone

    fake_models = types.ModuleType("torchvision.models")
    fake_models.ConvNeXt_Base_Weights = types.SimpleNamespace(IMAGENET1K_V1=concrete_weights)
    fake_models.convnext_base = fake_backbone
    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.models = fake_models
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", fake_models)
    encoder = ConvNeXtBaseEncoder("ConvNeXt_Base_Weights.IMAGENET1K_V1")
    features = encoder(torch.randn(2, 3, 32, 40))
    assert encoder.feature_dim == 13
    assert features.pooled.shape == (2, 13)
    assert features.tokens.shape == (2, 32 * 40, 13)
    assert calls == [concrete_weights]
    ConvNeXtBaseEncoder("none")
    assert calls[-1] is None
    with pytest.raises(ValueError, match="DEFAULT"):
        ConvNeXtBaseEncoder("DEFAULT")
