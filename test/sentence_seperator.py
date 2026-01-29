import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class TextCleaner:
    def __init__(self, model_name="HuggingFaceTB/SmolLM3-3B", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model: {model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have internet access to download the model or specify a local path.")
            raise e
            
        self.model.eval()
        
        self.punctuations = {
            ".": self.tokenizer.encode(".", add_special_tokens=False)[0],
            "?": self.tokenizer.encode("?", add_special_tokens=False)[0],
            "!": self.tokenizer.encode("!", add_special_tokens=False)[0]
        }
        
        self.space_token_ids = []
        test_tokens = self.tokenizer.encode(" ", add_special_tokens=False)
        if test_tokens:
            self.space_token_id = test_tokens[0]
        else:
            print("Warning: Could not find explicit space token. Space insertion might be degraded.")
            self.space_token_id = None

    def clean(self, raw_text, threshold_punct=0.1, threshold_space=0.1):
        
        inputs = self.tokenizer(raw_text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids[0]
        
        with torch.no_grad():
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits[0]  # [SeqLen, VocabSize]

        sentences = []
        current_sentence_tokens = []
        
        for i in range(len(input_ids)):
            curr_token_id = input_ids[i].item()
            current_sentence_tokens.append(curr_token_id)
            
            next_token_logits = logits[i]
            probs = F.softmax(next_token_logits, dim=-1)
            
            has_next = i < len(input_ids) - 1
            real_next_token_prob = probs[input_ids[i+1]].item() if has_next else 0.0

            should_insert_space = False
            if has_next and self.space_token_id is not None:
                space_prob = probs[self.space_token_id].item()
                if space_prob > threshold_space and space_prob > real_next_token_prob:
                    should_insert_space = True
            
            best_punct = None
            max_punct_prob = 0.0
            for char, pid in self.punctuations.items():
                p_prob = probs[pid].item()
                if p_prob > max_punct_prob:
                    max_punct_prob = p_prob
                    best_punct = pid
            
            if max_punct_prob > threshold_punct and max_punct_prob > real_next_token_prob:
                current_sentence_tokens.append(best_punct)
                
                decoded_sentence = self.tokenizer.decode(current_sentence_tokens, skip_special_tokens=True).strip()
                if decoded_sentence:
                    sentences.append(decoded_sentence)
                
                current_sentence_tokens = []
            elif should_insert_space:
                current_sentence_tokens.append(self.space_token_id)

        if current_sentence_tokens:
            decoded_sentence = self.tokenizer.decode(current_sentence_tokens, skip_special_tokens=True).strip()
            if decoded_sentence:
                sentences.append(decoded_sentence)

        return sentences
    
if __name__ == "__main__":
    cleaner = TextCleaner(model_name="HuggingFaceTB/SmolLM3-3B") # 가벼운 모델로 테스트

    # 테스트용 텍스트 (띄어쓰기, 문장부호 없음)
    text = """
    띄어쓰기와 문장 부호가 없는 날것의 텍스트를 정제하기 위해 아주 작은 언어모델을 정밀한 센서로 활용한 모델의 다음 토큰 예측 확률을 기반으로 매 시점마다 문맥적 종결감과 단어 간 결합도를 전수 조사한다 각 글자 사이에서 문장 부호나 공백이 나타날 확률을 계산하여 실제 다음 글자의 확률보다 높은 지점을 경계로 판정한다 특정 임계값을 넘는 지점에 마침표나 띄어쓰기를 삽입함으로써 문장을 자연스럽게 분리하고 교정한다 이는 생성 기능에 매몰되지 않고 언어모델의 내밀한 로짓값을 텍스트 정제 도구로 재해석한 효율적인 접근이다
    """.strip()

    print("\n[Original]:", text)
    cleaned = cleaner.clean(
        text, 
        threshold_punct=0.03,
        threshold_space=0.05,
    )
    print("[Cleaned] :", "\n", "\n".join(cleaned))