import torch
from diffusers import ZImagePipeline

class ZImageTurbo:
    def __init__(
        self, 
        model_name: str="Tongyi-MAI/Z-Image-Turbo",
    ):
        self.pipe = ZImagePipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda")

        self.pipe.transformer.set_attention_backend("flash")
        self.pipe.transformer.compile()
        self.pipe.enable_model_cpu_offload()
    
    def generate_image(
        self,
        prompt: str,
        height: int=1024,
        width: int=1024,
        num_inference_steps: int=9,
        guidance_scale: float=0.0,
        seed: int=42,
    ):
        generator = torch.Generator("cuda").manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        return image




if __name__ == "__main__":
    generator = ZImageTurbo()
    image = generator.generate_image("A serene landscape with mountains and a river during sunset.")
    image.save("generated_image.png")