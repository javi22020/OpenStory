from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

def load_model(model_path_or_url: str, device: str):
    pipe = StableDiffusionPipeline.from_single_file(model_path_or_url)
    pipe.to(device)
    return pipe

class ImageGenerator:
    def __init__(self, model_name) -> None:
        self.model = load_model(model_name)
    def change_model(self, model_name: str):
        self.model = load_model(model_name)
    def generate_image(self, prompt: str, size: tuple[int, int], steps: int, cfg: float):
        self.model(prompt)