from transformers import ViTModel

class ViTBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = ViTModel.from_pretrained("facebook/dino-vits16") if pretrained else ViTModel()
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state  # or outputs.pooler_output
