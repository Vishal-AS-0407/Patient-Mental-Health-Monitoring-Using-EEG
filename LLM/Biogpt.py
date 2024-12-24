import torch
from transformers import BioGptTokenizer, BioGptForCausalLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "microsoft/biogpt"
tokenizer = BioGptTokenizer.from_pretrained(model_name)
model = BioGptForCausalLM.from_pretrained(model_name).to(device)

def create_structured_prompt(question):
    prompt = f"""Based on the following scientific literature, provide a comprehensive analysis.

Question:
{question}

Requirements:
1. Analyze the interaction between arousal, valence, dominance, and error-related potentials
2. Explain their impact on emotional states,cognitive processes,desicision making skills,lerning capablities
3. Include practical implications for mental health cognitive performance,memory retention,emotional regulations
Detailed Analysis:
"""
    return prompt

def generate_inference_with_biogpt(question, min_length=900, max_length=1024):
    prompt = create_structured_prompt(question)

    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    generation_config = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'min_length': min_length,
        'max_length': max_length,
        'num_beams': 5, 
        'no_repeat_ngram_size': 3, 
        'length_penalty': 2.0,  
        'temperature': 0.7,  
        'top_p': 0.92,  
        'top_k': 50,  
        'repetition_penalty': 1.3,  
        'do_sample': True,
        'pad_token_id': tokenizer.eos_token_id,
        'early_stopping': True
    }
    
    with torch.no_grad():
        outputs = model.generate(**generation_config)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
  
    analysis = response.split("Detailed Analysis:")[-1].strip()
    
    return analysis

def main():
  
    question = """
    Analyze how high arousal, low valence, low dominance, and the presence of error-related potential affect emotional state, 
    cognitive load, decision-making, and learning capabilities. Specifically, how do these factors influence cognitive function, 
    memory, and decision-making, both in stressful and non-stressful situations?
    """
    
    print("Generating comprehensive analysis...")
    inference = generate_inference_with_biogpt(question)
    
    output_file = "biogpt.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("QUESTION:\n")
        file.write("-" * 80 + "\n")
        file.write(question + "\n\n")
        file.write("ANALYSIS:\n")
        file.write("-" * 80 + "\n")
        file.write(inference)
    
    print(f"\nDetailed analysis saved to {output_file}")

if __name__ == "__main__":
    main()
