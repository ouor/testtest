
import argparse
import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf

class Qwen3TTS:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        attn_backend: str = "flash_attention_2",
    ):
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is not available. A CUDA-capable GPU is required to run this model.")

        print(f"Loading model: {model_name}...")
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )

    def generate_prompt(
        self,
        ref_audio: str, # path to reference audio file
        ref_text: str, # text corresponding to reference audio
    ):
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        return prompt

    def generate_voice(
        self,
        text: str, # text to be synthesized
        language: str, # language of the text
        prompt, # voice clone prompt generated from generate_prompt()
        ):
        wav, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
        )
        return wav[0], sr

if __name__ == "__main__":
    tts = Qwen3TTS()

    prompt = tts.generate_prompt(
        ref_audio="test/ref.mp3", # path to reference audio file
        ref_text="아이.. 그게 참.. 난 정말 진심으로 말하고 있는거거든.. 요! 근데 그 쪽에서 자꾸 안 믿어주니까.." # text corresponding to reference audio
    )
    wav, sr = tts.generate_voice(
        "오전 10시 30분에 예정된 미팅 일정을 다시 한번 확인해 주시겠어요?", # text to be synthesized
        "Korean", # language of the text
        prompt # voice clone prompt generated from generate_prompt()
    )
    sf.write("generated_voice.wav", wav, sr)