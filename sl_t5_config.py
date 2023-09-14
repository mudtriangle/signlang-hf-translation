from transformers.models.t5 import T5Config


class SignLanguageT5Config(T5Config):
    def __init__(self, representations_channels=1024, n_patches_height=7, n_patches_width=7, adapter="conv", **kwargs):
        self.representations_channels = representations_channels
        self.n_patches_height = n_patches_height
        self.n_patches_width = n_patches_width

        self.adapter = adapter

        super().__init__(**kwargs)

