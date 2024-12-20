from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import torch.nn.functional as F
from .chunk_rag import extract_by_html2text_db_nolist, split_text_by_punctuation, extract_by_html2text_db_bench


def get_prob_subtract(model, tokenizer, sentence1, sentence2, language):
    if language == 'zh':
        query = '''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
        1. 将“{}”分割成“{}”与“{}”两部分；
        2. 将“{}”不进行分割，保持原形式；
        请回答1或2。'''.format(sentence1 + sentence2, sentence1, sentence2, sentence1 + sentence2)
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(
            query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids = prompt_ids
        output_ids = tokenizer.encode(['1', '2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()
        prob_subtract = next_token_prob_1 - next_token_prob_0
    else:
        query = '''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
        1. Split "{}" into "{}" and "{}" two parts;
        2. Keep "{}" unsplit in its original form;
        Please answer 1 or 2.'''.format(sentence1 + ' ' + sentence2, sentence1, sentence2, sentence1 + ' ' + sentence2)
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(
            query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids = prompt_ids
        output_ids = tokenizer.encode(['1', '2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()
        prob_subtract = next_token_prob_1 - next_token_prob_0
    return prob_subtract


def meta_chunking(original_text, base_model, language, ppl_threshold, chunk_length):
    model_name_or_path = 'Qwen/Qwen2.5-1.5B-Instruct'
    device_map = "auto"
    small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True,
                                                    cache_dir=r'/models')
    small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,
                                                       device_map=device_map,
                                                       cache_dir=r'/models')
    small_model.eval()
    chunk_length = int(chunk_length)
    if base_model == 'PPL Chunking':
        final_chunks = extract_by_html2text_db_nolist(original_text, small_model, small_tokenizer, ppl_threshold,
                                                      language=language)
    else:
        full_segments = split_text_by_punctuation(original_text, language)
        tmp = ''
        threshold = 0
        threshold_list = []
        final_chunks = []
        for sentence in full_segments:
            if tmp == '':
                tmp += sentence
            else:
                prob_subtract = get_prob_subtract(small_model, small_tokenizer, tmp, sentence, language)
                threshold_list.append(prob_subtract)
                if prob_subtract > threshold:
                    tmp += ' ' + sentence
                else:
                    final_chunks.append(tmp)
                    tmp = sentence
            if len(threshold_list) >= 5:
                last_ten = threshold_list[-5:]
                avg = sum(last_ten) / len(last_ten)
                threshold = avg
        if tmp != '':
            final_chunks.append(tmp)

    merged_paragraphs = []
    current_paragraph = ""
    if language == 'zh':
        for paragraph in final_chunks:
            if len(current_paragraph) + len(paragraph) <= chunk_length:
                current_paragraph += paragraph
            else:
                merged_paragraphs.append(current_paragraph)
                current_paragraph = paragraph
    else:
        for paragraph in final_chunks:
            if len(current_paragraph.split()) + len(paragraph.split()) <= chunk_length:
                current_paragraph += ' ' + paragraph
            else:
                merged_paragraphs.append(current_paragraph)
                current_paragraph = paragraph
    if current_paragraph:
        merged_paragraphs.append(current_paragraph)
    final_text = '\n\n'.join(merged_paragraphs)

    return final_text


def start_app():
    import gradio as gr
    with open('data/examples.json', 'r') as f:
        examples = json.load(f)
    original_prompt_list = [[s["original_text"]] for s in examples]

    title = "Meta-Chunking"

    header = """# Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception
            """

    theme = "soft"
    css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
                #anno-img .mask.active {opacity: 0.7}"""

    with gr.Blocks(title=title, css=css) as app:
        gr.Markdown(header)
        with gr.Row():
            with gr.Column(scale=3):
                original_text = gr.Textbox(value='', label="Original Text", lines=10, max_lines=10, interactive=True)
                chunking_result = gr.Textbox(value='', label="Chunking Result", lines=10, max_lines=10,
                                             interactive=False)

            with gr.Column(scale=1):
                base_model = gr.Radio(["PPL Chunking", "Margin Sampling Chunking"], label="Chunking Method",
                                      value="PPL Chunking", interactive=True)
                language = gr.Radio(["en", "zh"], label="Text Language", value="en", interactive=True)
                ppl_threshold = gr.Slider(minimum=0, maximum=1.0, step=0.1, value=0, label="Threshold",
                                          interactive=True)
                chunk_length = gr.Textbox(lines=1, label="Chunk length", interactive=True)
        button = gr.Button("⚡Click to Chunking")

        button.click(fn=meta_chunking,
                     inputs=[original_text, base_model, language, ppl_threshold, chunk_length],
                     outputs=[chunking_result])

        gr.Markdown("## Examples (click to select)")
        dataset = gr.Dataset(label="Meta-Chunking",
                             components=[gr.Textbox(visible=False, max_lines=3)],
                             samples=original_prompt_list,
                             type="index")

        dataset.select(fn=lambda idx: (
            examples[idx]["original_text"], examples[idx]["base_model"], examples[idx]["language"],
            examples[idx]["ppl_threshold"], examples[idx]["chunk_length"]),
                       inputs=[dataset],
                       outputs=[original_text, base_model, language, ppl_threshold, chunk_length])

    app.queue(max_size=10, api_open=False).launch(show_api=False, server_port=7080)


def get_chunk_tokens(text, tokenizer):
    tokens = tokenizer.encode(text, return_tensors='pt').shape[1]
    return tokens


def split_text_at_punctuation(text, tier1_punctuations, tier2_punctuations, tier3_punctuations):
    text_len = len(text)
    split_index = text_len // 3

    for i in range(split_index, split_index * 2):
        if text[i] in tier1_punctuations:
            return [text[:i + 1], text[i + 1:]]

    print('Warning: No tier1 punctuations found, you may want to check the text content. '
          'Trying harder splitters by default.')
    for i in range(split_index, split_index * 2):
        if text[i] in tier2_punctuations:
            return [text[:i + 1], text[i + 1:]]

    for i in range(split_index, split_index * 2):
        if text[i] in tier3_punctuations:
            return [text[:i + 1], text[i + 1:]]

    print('Error: No valid splitters found, default to split at the 1/3 point. '
          'You would want to check if the function is incompatible with the text content.'
          'Also, ensure that the text is not empty or too short for splitting.')

    return [text[0:split_index], text[split_index:]]


def dynamic_chunk(text, model, tokenizer, threshold, language, batch_size=4096):
    text_len = get_chunk_tokens(text, tokenizer)
    if text_len <= batch_size:
        return extract_by_html2text_db_nolist(text, model, tokenizer, threshold, language)
    else:
        return extract_by_html2text_db_bench(text, model, tokenizer,threshold, language, None, batch_size)


def ppl_chunking(text, max_token_size, threshold, ppl_model, using_tokenizer, language='zh'):
    assert ppl_model == 'Qwen/Qwen2.5-1.5B-Instruct'
    assert using_tokenizer == 'BAAI/bge-large-zh-v1.5'
    device_map = "auto"
    small_tokenizer = AutoTokenizer.from_pretrained(ppl_model, trust_remote_code=True,
                                                    cache_dir=r'/media/sata1/swx/models'
                                                    )
    small_model = AutoModelForCausalLM.from_pretrained(ppl_model, trust_remote_code=True,
                                                       device_map=device_map,
                                                       cache_dir=r'/media/sata1/swx/models',
                                                       torch_dtype=torch.bfloat16,
                                                       )
    small_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(using_tokenizer, cache_dir=r'/media/sata1/swx/models')
    chunks = [text]
    # recursively chunk the text until all chunks lengths are below the max_token_size
    while max([len(chunk) for chunk in chunks]) > max_token_size:
        longest_chunk_idx = max(range(len(chunks)), key=lambda x: len(chunks[x]))
        this_chunk = [chunks[longest_chunk_idx]]
        count = 3
        threshold = threshold
        while count > 0 and len(this_chunk) == 1:
            count -= 1
            this_chunk = dynamic_chunk(this_chunk[0], small_model, small_tokenizer, threshold,
                                       language)
            threshold -= 0.1
        if len(this_chunk) == 1:
            # perform hard split, find the first punctuation after 1/3 of the text
            text = this_chunk[0]
            tier1_punctuations = ['。', '！', '？', '；', '\n', '!', '?', ';']
            tier2_punctuations = ['，', '、', '：', ',', ':']
            tier3_punctuations = ['.', '(', ')', '[', ']', '{', '}', '（', '）', '【', '】', '「', '」', '『', '』', ' ']
            # split with lower tier punctuations if possible
            this_chunk = split_text_at_punctuation(text, tier1_punctuations, tier2_punctuations, tier3_punctuations)

        # insert back the split chunks inplace
        chunks.pop(longest_chunk_idx)
        chunks[longest_chunk_idx:longest_chunk_idx] = this_chunk

    # combine the chunks into bigger chunks that are below the max_token_size
    combined_chunks = []
    current_chunk = ""
    for chunk in chunks:
        if len(current_chunk) + len(chunk) <= max_token_size:
            current_chunk += chunk
        else:
            combined_chunks.append(current_chunk)
            current_chunk = chunk

    if current_chunk:
        combined_chunks.append(current_chunk)

    # format the chunks
    formatted_result = []
    for idx in range(len(combined_chunks)):
        formatted_result.append({
            "tokens": get_chunk_tokens(combined_chunks[idx], tokenizer),
            "content": combined_chunks[idx].strip(),
            "chunk_order_index": idx,
        })

    # release memory
    del small_model
    del small_tokenizer
    del tokenizer
    torch.cuda.empty_cache()

    return formatted_result

def print_memory_usage():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"Allocated Memory: {allocated / (1024 ** 2):.2f} MB")
    print(f"Reserved Memory: {reserved / (1024 ** 2):.2f} MB")


if __name__ == '__main__':
    device_map = "auto"
    small_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True,
                                                    cache_dir=r'/models')
    small_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True,
                                                       device_map=device_map,
                                                       cache_dir=r'/models')
    small_model.eval()

    print_memory_usage()

    import docx
    document_path = r'/datasets/公明镇去图片版本/公明镇 民政.docx'
    doc = docx.Document(document_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    result = extract_by_html2text_db_nolist(text, small_model, small_tokenizer, 0, language='zh')

    print_memory_usage()

    formatted_result = dict()
    for i, r in enumerate(result):
        formatted_result[i] = r
    output_path = r'/reference_code/Meta_Chunking/output/公明镇 民政.json'
    with open(output_path, 'w') as f:
        json.dump(formatted_result, f, ensure_ascii=False, indent=4)
