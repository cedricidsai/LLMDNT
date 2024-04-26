import os
import json
import openai 
import signal

path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)) + '/'

openai.api_key = os.getenv("OPENAI_API_KEY")
# Something else to try is to give the examples iteratively

storage_dir = "standard_prompting"

# model_name = "gpt-4"
model_name = "gpt-3.5-turbo"

storage_path = "../results/" + model_name + '/'

n_choices = 10

def read_dataset():
    dirs = os.listdir(path + "1D-ARC/dataset")
    file_paths = []
    for dir in dirs:
        file_paths += [path + "1D-ARC/dataset/" + dir + "/" + x for x in os.listdir(path + "1D-ARC/dataset/" + dir)]
    return(file_paths)

def generate_example(example):
    return("the original sequence \"{}\" is transformed into \"{}\"".format("".join([str(x) for x in example["input"][0]]), "".join([str(x) for x in example["output"][0]])))

def generate_prompt(data):
    prompt = []
    for example in data["train"]:
        prompt.append(generate_example(example))
    return ", ".join(prompt)

# def generate_test(example):
#     return("Please apply the rule to the following sequence and return only the transformed sequence : " + "".join([str(x) for x in example["input"][0]]))

# def verify_rule(messages, data):
#     systemprompt = "You are given the following rule to transform a sequence : \"{}\".".format(rule)
#     n_correct = 0
#     n_tests = 0
#     for example in data["train"]:
#         testprompt = generate_test(example)
#         test_truth = "".join([str(x) for x in example["output"][0]])
#         chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": systemprompt}, {"role": "user", "content": testprompt}])
#         test_response = chat_completion['choices'][0]['message']['content']
#         n_tests += 1
#         print(test_response)
#         print(test_truth)
#         if test_response == test_truth:
#             n_correct += 1
#         else:
#             return False

#     if n_correct == n_tests:
#         return True
#     return False


# def pass_test(rule, data):
#     systemprompt = "You are given the following rule to transform a sequence : \"{}\".".format(rule)

#     testprompt = generate_test(data["test"][0])
#     print(testprompt)
#     chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": systemprompt}, {"role": "user", "content": testprompt}])
#     test_response = chat_completion['choices'][0]['message']['content']
#     test_truth = "".join([str(x) for x in data["test"][0]["output"][0]])
#     print(test_response)
#     print(test_truth)
#     return(test_response == test_truth)

def generate_code(data, failed_code, temperature=0.5):
    systemprompt = "You are given the following sequences transitions and you are to find the pattern and write the code as a Python function \"transform(sequence)\" which transforms each original sequence into the transformed sequence. Respond with only the python function. Do not comment on the code."

    if failed_code:
        systemprompt += "Knowing that these functions failed : \n{}.".format("\n".join(failed_code))

    userprompt = generate_prompt(data)

    messages = [{"role": "system", "content": systemprompt}, {"role": "user", "content": userprompt}]
    chat_completion = openai.ChatCompletion.create(model=model_name, messages=messages, n=n_choices, temperature=temperature)
    response = chat_completion['choices'][0]['message']['content']
    messages += [{"role": "assistant", "content": response}]
    return(response, messages, chat_completion)

def handler(signum, frame):
    print("infinite loop")
    raise Exception("timeout")

def execute_verify(response, inputs, outputs):
    print(response["problem"])
    failed_code = []
    n_choice = 0 
    for choice in response["choices"]:
        n_choice += 1
        code = choice['message']['content']
        code = "\n".join([line for line in code.split('\n') if (line.startswith('def') or line.startswith('    ') or line.startswith('  '))])
        # print(code)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(1)

        try:
            import re
            scope = {'re':re}
            print(code)
            exec(code, scope)
            print("loaded")
            # print(code)
            # print(len(inputs))
            print(response["problem"], n_choice)
            if (response["problem"], n_choice) not in [('1d_fill_39',3)]:
                transformed = [scope['transform'](sequence) for sequence in inputs]
                if transformed == outputs:
                    return(True, code)
            # print(inputs)
            # print(transformed)
            # print(outputs)
            failed_code.append(code)
        except Exception as exc:
            print("code not running")
            print(exc)
            pass
        else:
            print("other exception")
            pass
            # print("failed loading")
        signal.alarm(0)
    return(False, failed_code)

def save_message(problem, messages, response_message):
    response_message["prompt"] = messages
    response_message["problem"] = problem
    with open(storage_path + storage_dir + "/{}.json".format(response_message["id"]), "w") as outfile:
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
    n_passed = 0
    n = 0
    for file in read_dataset():
        n += 1
        data = json.load(open(file))
        problem = file.split('.')[0].split('/')[-1]
        print_problem(problem, data)
        passed = False
        iteration = 0
        max_iterations = 1

        for response_file in os.listdir(storage_path + storage_dir + '/'):
            response = json.load(open(storage_path + storage_dir + '/{}'.format(response_file)))

            if response['problem'] == problem:
                valid, code = verify_response(response,  data["train"])
                iteration += 1
                if valid:
                    print('# code validates examples')
                    print(code)
                    test, code = verify_response(response,  data["test"])
                    if test:
                        print("# code passes test")
                        passed = True
                        n_passed += 1
                        break

        failed_code = []
        signal.alarm(0)
        while(not passed and iteration < max_iterations):
            iteration += 1
            rule, messages, response = generate_code(data, failed_code, temperature=1)
            # print(messages)
            save_message(problem, messages, response)
            valid, code = verify_response(response,  data["train"])
            if valid:
                print('# code validates examples')
                print(code)
                test, code = verify_response(response,  data["test"])
                if test:
                    print("# code passes test")
                    n_passed += 1
                    passed = True
                    break
            else:
                failed_code += code
        print("# passed : {}/{}".format(n_passed, n))


