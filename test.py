import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import os
import csv

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

 
output_csv = 'output.csv'


# Correct way to open one file for reading and another for writing
with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        
    # Create a CSV writer object to write to the output file
    writer = csv.writer(outfile)

    # Example: Writing headers from the input file to the output file
    writer.writerow(['image_filename', 'composition', 'unity_elements', 'unity_colors', 'balance_elements', 'balance_colors', 'elements', 'colors', 'movement', 'rhythm', 'focus_point', 'contrast_elements', 'contrast_colors', 'patterns', 'proportion', 'edgar_payne'])

    for image_filename in os.listdir("./images"):

        image_path = os.path.join("./images", image_filename)

        prompts = [
            "<image_placeholder>What can you say about the composition of this work?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the unity between the elements?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the unity between the colors?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the balance between the elements?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the balance between the colors?",
            "<image_placeholder>List all the elements that make up this painting.",
            "<image_placeholder>Describe all the elements that make up this painting.",
            "<image_placeholder>List all the colors used in this painting.",
            "<image_placeholder>Describe how the different colors are used in this painting.",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the movement depicted?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the rythm in the shown elements?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the focus point?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the contrast between the elements?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the contrast between the colors?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about patterns in the arrangement?",
            "<image_placeholder>On the subject of this paintings composition, what can you say about the proportion between the elements?",
            "<image_placeholder>On the subject of this paintings composition, how would you say it categorizes as Edgar Payne 15 archetypes of composition?",
        ]
        
        answers = []

        for prompt in prompts:

            conversation = [
                {
                    "role": "User",
                    "content": prompt,
                    "images": [image_path]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]

            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(vl_gpt.device)

            # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            print(f"{prepare_inputs['sft_format'][0]}", answer)

            answers.append(answer)

        writer.writerow([image_filename] + answers)

