from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel
import torch
import json
import re
import os


def load_model(model_path="finetuned_gpt2_lora"):
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' not found.")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base_model, model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, tokenizer


def format_prompt(input_data):
    example = (
        "Caliber: 0.308 in, Weight: 175 Gr, Length: 1.24 in, Muzzle Velocity: 2650 fps, "
        "Ballistic Coefficient: 0.453, Barrel Length: 20 in, Sight Height: 1.75 in, Twist Rate: 1:12 in, "
        "Zero Range: 100 yd, Temperature: 59 F, Altitude: 0 ft, Humidity: 50%, Pressure: 29.92 inHg, Wind Speed: 10 mph.\n"
        "Return the drop chart as JSON:\n"
        "[{\"distance\": 100, \"drop\": 0.0}, {\"distance\": 200, \"drop\": -3.2}, {\"distance\": 300, \"drop\": -11.5}]\n\n"
    )

    prompt = (
        example +
        f"Caliber: {input_data['caliber']}, Weight: {input_data['weight']}, Length: {input_data['length']}, "
        f"Muzzle Velocity: {input_data['muzzle_velocity']}, Ballistic Coefficient: {input_data['ballistic_coefficient']}, "
        f"Barrel Length: {input_data['barrel_length']}, Sight Height: {input_data['sight_height']}, Twist Rate: {input_data['twist_rate']}, "
        f"Zero Range: {input_data['zero_range']}, Temperature: {input_data['temperature']}, Altitude: {input_data['altitude']}, "
        f"Humidity: {input_data['humidity']}, Pressure: {input_data['pressure']}, Wind Speed: {input_data['wind_speed']}.\n"
        "Return the drop chart as JSON:\n"
    )
    return prompt



def predict_drop_chart(model, tokenizer, input_data):
    prompt = format_prompt(input_data)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            num_return_sequences=1,
            do_sample=False,
            # temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n===== RAW OUTPUT FROM MODEL =====\n", output_text)

    try:
        match = re.search(r"\[\s*{.*?}\s*\]", output_text, re.DOTALL)
        if not match:
            raise ValueError("Drop chart JSON not found in model output.")
        
        json_text = match.group(0)
        drop_chart = json.loads(json_text)
        return {"drop_chart": drop_chart}

    except Exception as e:
        print(f"Error parsing model output: {e}")
        return {"error": str(e)}
