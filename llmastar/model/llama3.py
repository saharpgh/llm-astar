import transformers
import torch

class Llama3:
  def __init__(self, device="cuda"):
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=0  # ← یا "cuda" به‌جای "cuda:3" برای Colab
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id
        ]

  
  def ask(self, prompt):
    outputs = self.pipeline(
      prompt,
      max_new_tokens=8000,
      eos_token_id=self.terminators,
      do_sample=False,
      temperature=None,
      top_p=None,
      pad_token_id=self.pipeline.tokenizer.eos_token_id
    )
    return outputs[0]["generated_text"][len(prompt):]
