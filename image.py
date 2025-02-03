from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model from Hugging Face
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use GPU for faster generation

# Define your prompt
prompt = "A futuristic cityscape at sunset, highly detailed, cyberpunk style"

# Generate the image
image = pipe(prompt).images[0]

# Save the image
image.save("generated_image.png")

print("Image generated and saved as 'generated_image.png'")