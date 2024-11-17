import json
import os
from consts import *
from llm_call import evoke_deepseek
from tqdm import tqdm

def read_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def get_input_prompt(question, ground, output):
    return ('下面给你一个问题和针对该问题的两个回答。第一个回答是基准答案，第二个回答是LLM生成的答案。'
            '请你对生成的答案进行评分，1是最低分，5是最高分。请注意，基准答案不一定是最佳答案。评价标准如下：'
            '5分：生成的答案完全包含基准答案的所有信息，并且包含的其他信息都与问题相关。\n'
            '4分：生成的答案包含基准答案的所有信息，但是有一些和问题不太相关的信息。\n'
            '3分：生成的答案仅包含基准答案的部分信息，但是没有错误信息。\n'
            '2分：生成的答案仅包含基准答案的部分信息，并且有一些错误信息。\n'
            '1分：生成的答案与基准答案完全不匹配，在时间/地点/人物等关键方面完全错误。\n'
            '当生成的答案包含基准答案的所有信息时，无论有多少额外信息，都应给不少于4分。\n'
            '在第一行中只告诉我评分，在第二行开始告诉我理由。\n'
            '回复举例： 1分 \\n 生成的答案与基准答案完全无关。\n'
            '问题：{}\n基准答案：{}\n生成的答案：{}'.format(question, ground, output))


def parse_unformatted_score_output(input_path, output_path):
    response = []
    temp_json = {'rating': None, 'reason': ''}

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            elif line.startswith('index'):
                if temp_json['rating'] is not None:
                    response.append(temp_json)
                    temp_json = {'rating': None, 'reason': ''}
            if line.endswith('分'):
                try:
                    temp_json['rating'] = int(line[0])
                except:
                    print(line)
            else:
                temp_json['reason'] += line
        response.append(temp_json)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=4)


def get_statistics(llm_output_parsed):
    parsed_info = json.load(open(llm_output_parsed, 'r', encoding='utf-8'))
    scores = [0] * 5
    for info in parsed_info:
        scores[info['rating'] - 1] += 1

    return scores




if __name__ == '__main__':
    idx = 4
    output_path, dataset_path = get_dataset_and_output(idx)
    dataset_name = dataset_path.split('\\')[-1]
    llm_output = read_json_lines(output_path)
    dataset = read_json_lines(dataset_path)

    save_path = fr'I:\python projects\api_call\RAG1.0\llm打分\{dataset_name}'
    with open(save_path, 'w', encoding='utf-8') as f:
        for i, (d, o) in enumerate(tqdm(zip(dataset, llm_output), total=len(dataset))):
            prompt = get_input_prompt(d['question'], d['answer'], o['answer'])
            f.write('index: ' + str(i) + '\n')
            f.write(evoke_deepseek(prompt) + '\n\n\n')

    i = os.path.join(llm_output_path, dataset_name)
    o = os.path.join(llm_output_parsed_path, dataset_name)
    parse_unformatted_score_output(i, o)

    print(get_statistics(o))


