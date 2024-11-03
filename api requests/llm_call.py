import json
import os

import openai
from openai import OpenAI
from tqdm import tqdm

from read_doc import *

DEFAULT_STEP = 70
DEFAULT_CHUNK_LEN = 80

qwen25_api_key = ""
deepseek_api_key = ""
gpt4o_api_key = ""


def generate_fuzzy_problems_with_qwen25(chunk, idx, output_path, output_filename):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=qwen25_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '我将给你一段地方志的片段，你的任务是根据该片段生成一个模糊问题。模糊问题指'
                                        '该问题虽然可以在文档中找到清晰的答案，但不涉及相关文档的任何关键词，在时间上的'
                                        '表述也不应具体到和文档一致。也就是说你应当模仿一位没有审阅过该地方志的用户进行'
                                        '提问。你的回答格式应为：{你提出的问题} \\n \\n {该问题的简短答案} \\n \\n {地方志中'
                                        '对该问题的相关描述,用序号分离不同片段}。你不应在引用地方志原文的过程中更改任何信息。'
                                        f'下面是地方志的片段：{chunk}'}],
    )

    # collect the generated problems and store them in the path, use append mode
    with open(os.path.join(output_path, output_filename), 'a') as f:
        # write the corresponding index
        f.write(f'Index: {idx}\n')
        # write the model output

        json_data = completion.model_dump_json()
        # Check if parsing is needed
        if isinstance(json_data, str):
            json_data = json.loads(json_data)  # Parse if it's a string
        answers = json_data['choices'][0]['message']['content'] + '\n\n'
        f.write(answers)


def generate_fuzzy_problems_with_deepseek(chunk, idx, output_path, output_filename):
    from openai import OpenAI

    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '我将给你一段地方志的片段，你的任务是根据该片段生成一个模糊问题。模糊问题指'
                                        '该问题虽然可以在文档中找到清晰的答案，但不涉及相关文档的任何关键词，在时间上的'
                                        '表述也不应具体到和文档一致。也就是说你应当模仿一位没有审阅过该地方志的用户进行'
                                        '提问。你的回答格式应为：{你提出的问题} \\n \\n {该问题的简短答案} \\n \\n {地方志中'
                                        '对该问题的相关描述,用序号分离不同片段}。你不应在引用地方志原文的过程中更改任何信息。'
                                        f'下面是地方志的片段：{chunk}'},
        ],
        stream=False
    )

    # collect the generated problems and store them in the path, use append mode
    with open(os.path.join(output_path, output_filename), 'a') as f:
        # write the corresponding index
        f.write(f'Index: {idx}\n')
        # write the model output

        json_data = response.model_dump_json()
        # Check if parsing is needed
        if isinstance(json_data, str):
            json_data = json.loads(json_data)  # Parse if it's a string
        answers = json_data['choices'][0]['message']['content'] + '\n\n'
        f.write(answers)


## deprecated, only for testing, cost twice as much
def generate_fuzzy_problems_with_gpt4o(chunk, idx, output_path, output_filename):
    from openai import OpenAI

    # import os
    # os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    # os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    # os.environ['no_proxy'] = '127.0.0.1,localhost'
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ['NO_PROXY'] = '127.0.0.1,localhost'

    client = OpenAI(
        api_key=gpt4o_api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '我将给你一段地方志的片段，你的任务是根据该片段生成一个模糊问题。模糊问题指'
                                        '该问题虽然可以在文档中找到清晰的答案，但不涉及相关文档的任何关键词，在时间上的'
                                        '表述也不应具体到和文档一致。也就是说你应当模仿一位没有审阅过该地方志的用户进行'
                                        '提问。你的回答格式应为：{你提出的问题} \\n \\n {该问题的简短答案} \\n \\n {地方志中'
                                        '对该问题的相关描述,用序号分离不同片段}。你不应在引用地方志原文的过程中更改任何信息。'
                                        f'下面是地方志的片段：{chunk}'},
        ]
    )

    # collect the generated problems and store them in the path, use append mode
    with open(os.path.join(output_path, output_filename), 'a') as f:
        # write the corresponding index
        f.write(f'Index: {idx}\n')
        # write the model output

        json_data = response.model_dump_json()
        # Check if parsing is needed
        if isinstance(json_data, str):
            json_data = json.loads(json_data)  # Parse if it's a string
        answers = json_data['choices'][0]['message']['content'] + '\n\n'
        f.write(answers)

    f.close()


def generate_batch_job(doc_path, output_path):
    doc_name = os.path.basename(doc_path)
    if doc_name.endswith('.doc') or doc_name.endswith('.docx') or doc_name.endswith('.txt'):
        doc_name = doc_name.split('.')[0]
    doc_length = find_file_length(doc_path)

    print(f"Document name: {doc_name} \n Document length: {doc_length} \n Generating tasks...")

    batch_data = []
    for start_idx in tqdm(range(0, doc_length - DEFAULT_CHUNK_LEN, DEFAULT_STEP)):
        idx = slice(start_idx, start_idx + DEFAULT_CHUNK_LEN)
        chunk = read_docx(doc_path, idx)
        task = {
            "custom_id": f"task-{doc_name}-{start_idx}-{DEFAULT_CHUNK_LEN}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                # This is what you would have in your Chat Completions API call
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": 'You are a helpful assistant.'
                    },
                    {
                        "role": "user",
                        "content": '我将给你一段地方志的片段，你的任务是根据该片段生成一个模糊问题。模糊问题指该问'
                                   '题虽然可以在文档中找到清晰的答案，但不涉及相关文档的任何关键词，在时间上的表述'
                                   '也不应具体到和文档一致。也就是说你应当模仿一位没有审阅过该地方志的用户进行提问。'
                                   '你的回答格式应为：{你提出的问题} \\n \\n {该问题的简短答案} \\n \\n {地方志中'
                                   '对该问题的相关描述,用序号分离不同片段}。你不应在引用地方志原文的过程中更改任何信息。'
                                   f'下面是地方志的片段：{chunk}'
                    }
                ],
            }
        }
        batch_data.append(task)

    print("Task generation finished, writing to file...")

    # encoding must support chinese characters
    with open(os.path.join(output_path, doc_name + '.jsonl'), 'w') as f:
        for task in batch_data:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')

    f.close()


def upload_batch_job(doc_name, input_path, response_output_path):
    '''
    :param doc_name: document name
    :param input_path: where the task file is stored (the output path of generate_batch_job)
    :param response_output_path: where the batch job id and response file will be stored

    !! store your batch job id safely so that you can retrieve the results later
    '''
    client = openai.Client(api_key=gpt4o_api_key)
    print('Uploading batch job...')
    upload_trials = 0
    while upload_trials < 3:
        try:
            batch_instance = client.files.create(
                file=open(os.path.join(input_path, doc_name + '.jsonl'), 'rb'),
                purpose='batch',
            )
            break
        except Exception as e:
            print(e)
            upload_trials += 1
    else:
        print('Failed to upload batch job after 3 trials, terminating...')
        return

    print('Successfully uploaded batch job')
    print(batch_instance)
    print('-' * 50)

    print('Creating batch job...')
    create_trials = 0
    while create_trials < 3:
        try:
            batch_job = client.batches.create(
                input_file_id=batch_instance.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            break
        except Exception as e:
            print(e)
            create_trials += 1
    else:
        print('Failed to create batch job after 3 trials, terminating...')
        return

    print('Successfully created batch job')
    print(batch_job.id)
    with open(os.path.join(response_output_path, 'batch_jobs.txt'), 'a') as f:
        f.write(batch_job.id + '\n')
    f.close()
    print('-' * 50)
    print('Upload finished')


def retrieve_batch_job(batch_job_id, doc_name, response_output_path):
    client = openai.Client(api_key=gpt4o_api_key)

    print('Retrieving batch job...')
    retrieve_trials = 0
    while retrieve_trials < 3:
        try:
            batch_job = client.batches.retrieve(batch_job_id)
            break
        except Exception as e:
            print(e)
            retrieve_trials += 1
    else:
        print('Failed to retrieve batch job after 3 trials, terminating...')
        return

    print('Successfully retrieved batch job')
    print(batch_job)
    print('-' * 50)

    if batch_job.status != 'completed':
        print('Batch job not completed, need to wait...')
        return
    else:
        print('Batch job already completed, retrieving results...')
        result_id = batch_job.output_file_id
        file_retrieve_trials = 0
        while file_retrieve_trials < 3:
            try:
                result_file = client.files.content(result_id)
                break
            except Exception as e:
                print(e)
                file_retrieve_trials += 1
        else:
            print('Failed to retrieve result file after 3 trials, terminating...')
            return

        # content = result_file.content.decode('utf-8')
        # with open(os.path.join(response_output_path, doc_name + '_dataset.txt'), 'w') as f:
        #     f.write(content)
        #
        # print('Results retrieved and stored in file')

        # Assuming result_file.content is the raw JSON response in binary form
        raw_content = result_file.content.decode('utf-8')  # Decode binary to string first

        output_path = os.path.join(response_output_path, doc_name + '_dataset.txt')
        for line in raw_content.splitlines():
            parsed_content = json.loads(line)
            content_text = parsed_content['response']['body']['choices'][0]['message']['content']

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(content_text + '\n\n\n')

        print('Results retrieved and stored in file')



if __name__ == '__main__':
    doc_path = r'xxx\光明街道年鉴2022.docx'
    doc_name = '光明街道年鉴2022'
    batch_data_output_path = r'xxx\batch_data'
    response_output_path = r'xxx\response'
    # generate_batch_job(doc_path, batch_data_output_path)
    # upload_batch_job(doc_name, batch_data_output_path, response_output_path)
    retrieve_batch_job('some_batch_id', doc_name, response_output_path)
