"""Optional synthetic model checks. Set CCTA_CHAT_BACKEND to the patched checkout."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

BACKEND = os.environ.get("CCTA_CHAT_BACKEND")
AVAILABLE = bool(BACKEND and importlib.util.find_spec("peft"))


@unittest.skipUnless(AVAILABLE, "optional: install model extra and set CCTA_CHAT_BACKEND")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ccta_chat.backend import load_backend
        torch.set_num_threads(2)
        cls.module = load_backend(BACKEND)

    def model(self):
        import torch
        torch.manual_seed(7)
        config = self.module.LlavaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                                        mm_vision_tower="cached-ctvit", mm_hidden_size=512, mm_context_size=512,
                                        mm_projector_type="attn_pool+mlp2x_gelu", mm_patch_merge_type="flat",
                                        image_aspect_ratio="square", pad_token_id=0, bos_token_id=1, eos_token_id=2)
        return self.module.LlavaLlamaForCausalLM(config)

    def test_visual_insertion_and_feature_influence(self):
        import torch
        model = self.model().eval()
        ids = torch.tensor([[1, -200, 3, 4]])
        images = torch.randn(1, 1, 2, 2, 512)
        labels = torch.tensor([[-100, -100, 3, 4]])
        inputs = model.prepare_inputs_labels_for_multimodal(ids, None, torch.ones_like(ids), None, labels, images)
        self.assertEqual(inputs[4].shape, (1, 259, 32))
        self.assertEqual(int(inputs[5].eq(-100).sum()), 257)
        with torch.no_grad():
            first = model(ids, images=images).logits
            second = model(ids, images=torch.zeros_like(images)).logits
            text = model(torch.tensor([[1, 3, 4]])).logits
        self.assertGreater(float((first-second).abs().max()), 1e-6)
        self.assertEqual(text.shape, (1, 3, 32))

    def test_gradients_and_trainable_adapter_restore(self):
        import torch
        from peft import get_peft_model_state_dict
        from ccta_chat.model import initialize_adapter, trainable_components, adapter_settings
        settings = {"lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0.0}
        model = initialize_adapter(self.model(), mode="ft1", settings=settings)
        counts = trainable_components(model)
        self.assertGreater(counts["lora"], 0)
        self.assertGreater(counts["projector"], 0)
        ids = torch.tensor([[1, -200, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])
        features = torch.randn(1, 1, 2, 2, 512)
        before = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        loss = model(ids, images=features, labels=labels).loss
        loss.backward()
        for marker in ("lora_", "mm_projector"):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for n,p in model.named_parameters() if marker in n))
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.05).step()
        for marker in ("lora_", "mm_projector"):
            self.assertTrue(any(not p.detach().equal(before[n]) for n,p in model.named_parameters()
                                if marker in n))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = initialize_adapter(self.model(), mode="ft2", settings={"lora_rank": 2}, adapter=directory)
            left, right = get_peft_model_state_dict(model), get_peft_model_state_dict(restored)
            self.assertTrue(all(left[k].detach().cpu().equal(right[k].detach().cpu()) for k in left))
            self.assertTrue(all(p.requires_grad for n,p in restored.named_parameters() if "lora_" in n))
            with self.assertRaisesRegex(ValueError, "rank"):
                adapter_settings(directory, expected_rank=64)

    def test_mixed_batch_preserves_text_first_slot(self):
        import torch
        from ccta_chat.training import collate
        text = {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([-100, 2])}
        image = {"input_ids": torch.tensor([1, -200, 3]), "labels": torch.tensor([-100, -100, 3]),
                 "image": torch.ones(1, 2, 2, 512)}
        result = collate([text, image], pad_token_id=2)
        self.assertEqual(result["images"].shape, (2, 1, 2, 2, 512))
        self.assertEqual(float(result["images"][0].sum()), 0)
        self.assertEqual(result["attention_mask"].tolist(), [[True, True, False], [True, True, True]])

    def test_preprocess_axis_order_and_padding(self):
        import numpy as np
        from ccta_chat.encoding import preprocess
        result = preprocess(np.zeros((4, 6, 2), dtype=np.float32), (1, 1, 1),
                            target_spacing=(1, 1, 1), shape=(6, 8, 4))
        self.assertEqual(result.shape, (1, 1, 4, 6, 8))
        self.assertAlmostEqual(float(result[0,0,1,2,2]), 1/9)
        self.assertEqual(float(result[0,0,0,0,0]), -1)

    def test_encoder_reuse_requires_matching_provenance(self):
        import types
        import numpy as np
        import nibabel as nib
        import torch
        from unittest.mock import patch
        from ccta_chat.encoding import encode
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"source"
            (source/"transformer_maskgit").mkdir(parents=True)
            module_file = source/"transformer_maskgit/__init__.py"
            module_file.write_text("# Independent synthetic encoder stub\n")
            weights = root/"encoder.pth"
            weights.write_bytes(b"independent test fixture")
            volume = nib.Nifti1Image(np.zeros((2,2,2), dtype=np.float32), np.diag([.5,.5,1.,1.]))
            nib.save(volume, root/"input.nii.gz")
            class Encoder(torch.nn.Module):
                def __init__(self, **kwargs): super().__init__()
                def load(self, path): pass
                def forward(self, tensor, return_encoded_tokens=False):
                    return torch.ones(1,1,2,2,512)
            module = types.ModuleType("transformer_maskgit")
            module.__file__ = str(module_file)
            module.CTViT = Encoder
            with patch.dict("sys.modules", {"transformer_maskgit": module}), \
                 patch("ccta_chat.encoding.preprocess", return_value=torch.ones(1,1,2,2,2)) as preprocessing:
                result = encode(root/"input.nii.gz", root/"output.npz", source, weights, device="cpu")
                self.assertFalse(result["reused"])
                self.assertEqual(preprocessing.call_args.args[1], (.5,.5,1.))
                self.assertTrue(encode(root/"input.nii.gz", root/"output.npz", source, weights,
                                       reuse=True, device="cpu")["reused"])
                with self.assertRaisesRegex(ValueError, "provenance"):
                    encode(root/"input.nii.gz", root/"output.npz", source, weights,
                           spacing=(1,1,1), reuse=True, device="cpu")
                (root/"output.npz").write_bytes(b"corrupted cached output")
                with self.assertRaisesRegex(ValueError, "provenance"):
                    encode(root/"input.nii.gz", root/"output.npz", source, weights, reuse=True, device="cpu")


if __name__ == "__main__":
    unittest.main()
