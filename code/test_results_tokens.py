import os
import json
import openai 
import signal
import pandas as pd
import tiktoken
from collections import defaultdict 

path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)) + '/'

storage_dir = ''

storage_path = '../results/'

token_encoding = tiktoken.encoding_for_model('gpt-4')

max_choices = 10

n_choices = 1

def read_dataset():
    dirs = os.listdir(path + "1D-ARC/dataset")
    file_paths = []
    for dir in dirs:
        file_paths.append((dir, [path + "1D-ARC/dataset/" + dir + "/" + x for x in os.listdir(path + "1D-ARC/dataset/" + dir)]))
    return(file_paths)

def generate_code(data, failed_code, temperature=0.5):
    systemprompt = "You are given the following sequences transitions and you are to find the pattern and write the code as a Python function \"transform(sequence)\" which transforms each original sequence into the transformed sequence. Respond with only the python function. Do not comment on the code."

    if failed_code:
        systemprompt += "Knowing that these functions failed : \n{}.".format("\n".join(failed_code))

    userprompt = generate_prompt(data)

    messages = [{"role": "system", "content": systemprompt}, {"role": "user", "content": userprompt}]
    chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=messages, n=5, temperature=temperature)
    response = chat_completion['choices'][0]['message']['content']
    messages += [{"role": "assistant", "content": response}]
    return(response, messages, chat_completion)

def handler(signum, frame):
    print("infinite loop")
    raise Exception("timeout")

def execute_verify(response, inputs, outputs):
    # print(response["problem"])
    failed_code = []
    n_choice = 0 
    for choice in response["choices"][:n_choices]:
        n_choice += 1
        code = choice['message']['content']
        code = "\n".join([line for line in code.split('\n') if ((line.startswith('def') or line.startswith('    ') or line.startswith('  ')) and (not line.startswith('     -')))])
        # print(code)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(1)

        try:
            import re
            scope = {'re':re}
            # print(code)
            exec(code, scope)
            # print("loaded")
            # print(code)
            # print(len(inputs))

            print("('%s','%s',%i)"%(storage_dir, response["problem"], n_choice))
            if (storage_dir, response["problem"], n_choice) not in [('standard_prompting', '1d_fill_39',3), ('chain_of_thought','1d_pcopy_mc_28',5), ('chain_of_thought','1d_pcopy_1c_4',1)]:
                transformed = [scope['transform'](sequence) for sequence in inputs]
                if transformed == outputs:
                    return(True, code)
            # print(inputs)
            # print(transformed)
            # print(outputs)
        except Exception as exc:
            # print("code not running")
            # print(exc)
            pass
        else:
            # print("other exception")
            pass
            # print("failed loading")
        failed_code.append(code)
        signal.alarm(0)
    return(False, failed_code)

def save_message(problem, messages, response_message):
    response_message["prompt"] = messages
    response_message["problem"] = problem
    with open(storage_dir + "/{}.json".format(response_message["id"]), "w") as outfile:
        json.dump(response_message, outfile)


def verify_response(response, examples):
    test_sequences = ["".join([str(x) for x in example["input"][0]]) for example in examples]
    test_truths = ["".join([str(x) for x in example["output"][0]]) for example in examples]
    valid, code = execute_verify(response, test_sequences, test_truths)
    return(valid, code)

def print_problem(problem, data):
    print("# {}".format(problem))
    test_sequences = ["".join([str(x) for x in example["input"][0]]) for example in data["train"]]
    test_truths = ["".join([str(x) for x in example["output"][0]]) for example in data["train"]]
    print(' input  : ', test_sequences)
    print(' output : ', test_truths)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        exit()
    storage_dir = sys.argv[1]
    print("testing : ", storage_dir)

    saved_data = pd.DataFrame()

    for current_choices in range(1, max_choices+1):
        n_choices = current_choices
        n_passed = 0
        categories_seen = defaultdict(int)
        categories_passed = defaultdict(int)
        n = 0
        # print(read_dataset())

        input_tokens = 0
        output_tokens = 0

        for (category, files) in read_dataset():
            for file in files:
                categories_seen[category] += 1
                n += 1
                data = json.load(open(file))
                problem = file.split('.')[0].split('/')[-1]
                # print_problem(problem, data)
                passed = False
                iteration = 0
                max_iterations = 1
                for response_file in os.listdir(storage_path + storage_dir + '/'):
                    signal.alarm(0)
                    response = json.load(open(storage_path + storage_dir + '/{}'.format(response_file)))

                    if response['problem'] == problem:
                        valid, code = verify_response(response,  data["train"])
                        iteration += 1
                        input_tokens += response["usage"]["prompt_tokens"]

                        output_tokens += sum([len(token_encoding.encode(choice["message"]["content"])) for choice in response["choices"][:n_choices]]) # response["usage"]["completion_tokens"]

                        if valid:
                            # print('# code validates examples')
                            # print(code)
                            test, code = verify_response(response,  data["test"])
                            if test:
                                # print("# code passes test")
                                passed = True
                                n_passed += 1
                                categories_passed[category] += 1
                                break
                print(n, n_passed)

        signal.alarm(0)

        output_tokens = current_choices * output_tokens / max_choices
        data = pd.DataFrame(categories_passed, index = [storage_dir])
        data = data.assign(choices = [current_choices])
        data = data.assign(input_tokens = [input_tokens])
        data = data.assign(output_tokens = [output_tokens])
        saved_data = pd.concat([saved_data, data])

    saved_data.to_csv(storage_path + storage_dir + '_tokens.csv')

    # categories_names = "Move 1,Move 2,Move 3,Move Dynamic,Move 2 Towards,Fill,Padded Fill,Hollow,Flip,Mirror,Denoise,Denoise Multicolor,Pattern Copy,Pattern Copy Multicolor,Recolor by Odd Even,Recolor by Size,Recolor by Size Comparison,Scaling".split(',')
    # categories_dirs = "1d_move_1p,1d_move_2p,1d_move_3p,1d_move_dp,1d_move_2p_dp,1d_fill,1d_padded_fill,1d_hollow,1d_flip,1d_mirror,1d_denoising_1c,1d_denoising_mc,1d_pcopy_1c,1d_pcopy_mc,1d_recolor_oe,1d_recolor_cnt,1d_recolor_cmp,1d_scale_dp".split(',')

    # print(categories_passed)
    # print(categories_seen)
    # print(n, n_passed)
    # print()

    # print("\\begin{tabular}{|l|c|c|}")
    # print("\hline")
    # print("Task category name & solved & tested \\\\ \hline")
    # for i, category in enumerate(categories_dirs):
    #     print(categories_names[i], '&', categories_passed[category], '&', categories_seen[category], "\\\\ \hline")
    # print("\\end{tabular}")    
