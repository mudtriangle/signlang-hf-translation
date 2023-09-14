from transformers.models.whisper import WhisperConfig


class SignLanguageWhisperConfig(WhisperConfig):
    def __init__(self, representations_channels=1024, n_patches_height=7, n_patches_width=7, **kwargs):
        self.representations_channels = representations_channels
        self.n_patches_height = n_patches_height
        self.n_patches_width = n_patches_width

        super().__init__(**kwargs)

