"""
Functions to process inputs, make calls to the Open AI API, and process outputs from the API.
"""
import openai
import threading
from tiktoken import encoding_for_model
from transformers import AutoTokenizer

from constants import *

#constants/default values
MODEL = 'text-davinci-002' # default model.
TEMP = 0
MAX_TOKENS = 500
BEST_OF = 1
LOGPROBS = 1 # number of tokens to return logprobability for. maximum 5. (not currently supported for chat models)
WAIT = 2 # seconds to wait between each call to the api, to avoid rate limit
TIMEOUT = 40 # seconds to wait (per question) before timeout (mostly handled in threads, not in API calls)

# set Huggingface auth token. You should request for Meta's llama-3-70b on Huggingface and use the auth token here after you are granted access
hf_auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

# set api key. You can manually paste it in here or set it as an environmental variable
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
openai.api_key = API_KEY
if BASE_URL:
    openai.base_url = BASE_URL


CHAT_MODELS = ['gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-4-32k-0613', 'gpt-3.5-turbo',
               'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613',
               'gpt-4-1106-preview', 'meta-llama/Meta-Llama-3-70B-Instruct']
RATE_LIMITS = {'gpt-4': (200, 40000),
               'chat': (3500, 90000),
               'text': (3500, 350000)} # rate limits for different models, in requests and tokens per minute

def parse_base_model_name(model_name: str) -> str:
    """
    Using regex to capture the base model name from a fine-tuned model.
    """
    match = re.search(r'(gpt-[^\s:]+)', model_name)
    if match:
        return match.group(1)
    return model_name

def batch_inputs (questions:list, model:str, verbose:bool, thread_num:int=1):
    """
    Group questions into batches.
    Batch sizes are determined by token costs, to keep under rate limit for model, or by number of threads.

    Args:
    - questions: model inputs to be batched
    - model: name of the model in OpenAI API (for identifying rate limits)
    - verbose: whether the process prints to the console

    Keyword Args:
    - thread_num: number of threads to plan for. If thread_num is provided, use that number of batches.

    Returns: a list of batches, where each batch is a list of strings.
    """
    if isinstance(questions, str): # if only one input is passed, return it as a batch of 1
        return [[questions]]
    batches = []
    token_limit = 0
    base = parse_base_model_name(model)
    if base == 'meta-llama/Meta-Llama-3-70B-Instruct':
        tokenizer = AutoTokenizer.from_pretrained(base, use_auth_token=hf_auth_token)
    else:    
        tokenizer = encoding_for_model(base)
    chat = True
    if 'gpt-4' in model:
        request_limit, token_limit = RATE_LIMITS['gpt-4']
    elif base in CHAT_MODELS:
        request_limit, token_limit = RATE_LIMITS['chat']
    else:
        request_limit, token_limit = RATE_LIMITS['text']
        chat = False

    if thread_num > 1: # multi-thread: ignore rate limits and make 1 batch per thread
        if verbose: print("Number of threads:", thread_num)
        batches = [[] for i in range(thread_num)]
        for i, question in enumerate(questions):
            batches[i%thread_num].append(question)
    elif chat:
        return questions # if the model is a chat model, and there's only 1 thread, keep each question in its own batch.
    else: # single thread text model
        current_batch = []
        current_batch_tokens = 0
        token_cost = 0
        for i, question in enumerate(questions):
            if (current_batch_tokens + token_cost >= token_limit): # current question does not fit in current batch
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0

            question_tokens = len(tokenizer.encode(question))
            token_cost = question_tokens + MAX_TOKENS
            current_batch.append(question)
            current_batch_tokens += token_cost
        batches.append(current_batch)

    if verbose: print(f"Split {len(questions)} inputs into {len(batches)} batch{'es' if len(batches) > 1 else ''}.")
    return batches

def build_messages(question:str, split_answer:bool=True, system_prompt:str='', post_reasoning_prompt:str=''):
    """
    Format the question into a list of messages, compatible with the OpenAI ChatCompletion API.
    """
    messages = []
    if system_prompt:
        messages += [{"role":"system","content": system_prompt}]
    if split_answer and 'A:' in question:
        just_question, reasoning = question.split('A:', 1)
        just_question = just_question.strip()
        reasoning = 'A: '+reasoning.replace("A:", "").strip()
        messages += [{"role":"user","content": just_question}, {"role":"assistant","content": reasoning}]
    else:
        messages += [{"role":"user","content": question}]
    if post_reasoning_prompt:
        messages.append({"role":"user", "content":post_reasoning_prompt})
    return messages

def messages_to_str(messages):
    """
    Convert messages object into printable string.
    """
    contents = [message['content'] for message in messages]
    if len(contents) > 2: # if the messages is long, just print the last message.
        contents = contents[-1:]
    return '\n\n'.join(contents)

def query_model(batches:list, model:str, echo:bool = False, max_length:int = MAX_TOKENS,
                thread_num:int = 1, results:dict = {}, post_reasoning_prompt:str = '', temp=TEMP, split_answer=True,
                system_prompt='', timeout=TIMEOUT):
    """
    Handle queries to the OpenAI models.

    Args:
    - batches: batches of inputs (generated by batch_inputs)
    - model: name of the model in OpenAI API

    Keyword Args:
    - echo: whether prompt is repeated in response text
    - max_length: max number of tokens in text completion
    - thread_num: number of threads to use for text generation.
        - If thread_num > 1, uses threading library to perform multithreading, with each thread handling a batch.
    - results: dictionary of prior results.
        - This is mutated during each function call, to allow for aggregating information from inside threads.
    - post_reasoning_prompt: prompt text for after reasoning (i.e. for text repetition task)

    Returns: dictionary of results, where keys are questions, and values are tuples containing:
        (generated text, total logprobability of generation, full API response)
        - Logprobability value is None for chat models.
    """
    if thread_num > 1: # multi-threaded
        thread_list = []
        for i, batch in enumerate(batches):
            if batch: # do not create threads for empty batches
                thread_list.append((threading.Thread(target=query_model,
                                                        args=(batch, model),
                                                        kwargs={"echo":echo,
                                                                "max_length":max_length,
                                                                "thread_num":1,
                                                                "results":results,
                                                                'post_reasoning_prompt':post_reasoning_prompt,
                                                                'system_prompt':system_prompt}),
                                    batch)) # store results for the thread by mutating these lists

                print(f"{thread_list[-1][0].name}: {len(batch)} question{'' if len(batch) == 1 else 's'}")
        # Start every thread
        for thread, batch in thread_list:
            thread.start()
        # Wait for every thread to end
        for thread, batch in thread_list:
            thread.join(timeout=timeout*len(batch)) # allow timeout per question.
            if thread.is_alive():
                print(f"{thread.name} timed out.")
                continue # allow timed-out threads to run until all others have been checked
            print(f"{thread.name} finished.")
        for thread, batch in thread_list: # final check for straggler threads
            if thread.is_alive():
                print(f"FINAL CHECK - {thread.name} timed out.")
        print("All threads finished")
    elif parse_base_model_name(model) in CHAT_MODELS: # single thread - chat model
        for question in batches: # pass questions to model 1 at a time
            if not question: # ignore falsy (i.e. empty string, None) values
                continue
            if isinstance(question, dict) and len(question) == 1: # messages object and question key already provided
                question, messages = list(question.items())[0]
            elif isinstance(question, list): # messages object already provided
                messages = question
                question = messages[-1]['content'] # index results by the last message content
            else: # format messages
                messages = build_messages(question, split_answer=split_answer, system_prompt=system_prompt,
                                          post_reasoning_prompt=post_reasoning_prompt)
            time.sleep(WAIT)
            try:
                response = openai.ChatCompletion.create(messages=messages,
                model=model,
                temperature = temp,
                max_tokens= max_length
                )

                # put the single answer into the batch records
                response_text = response.choices[0]['message']['content'].strip()
            except Exception as e: # if there was an exception in this question, return None values.
                print(e)
                response = None
                response_text = None
            results[question] = (response_text,
                                None, # logprobs are not currently supported by the ChatCompletion API.
                                response) # full response
            print('~~~~~~~~~~~~\n\nPrompt:', messages_to_str(messages)+'~~~')
            print('Answer:', response_text)
    else: # single thread - text model
        for batch in batches:
            if post_reasoning_prompt:
                batch = [q+post_reasoning_prompt for q in batch if q]
            else:
                batch = [q for q in batch if q] # ignore falsy (i.e. empty string, None)
            time.sleep(WAIT)
            try:
                response = openai.Completion.create(prompt=batch,
                model=model,
                temperature = temp,
                max_tokens=max_length,
                best_of = BEST_OF,
                logprobs = LOGPROBS,
                echo = echo
                )

                for choice in response.choices:
                    # order of choices is not necessarily order of prompt, so need to use index attribute to match
                    print('~~~~~~~~~~~~\n\nPrompt:', batch[choice.index]+'~~~')
                    print('Answer:', choice['text'].strip())
                    results[batch[choice.index]] = (choice['text'].strip(),
                                                    sum(choice['logprobs']['token_logprobs'], response))
            except Exception as e:
                print(e)
                for question in batch: # if there was an exception in this batch, return None values.
                    results[question] = (None,None,None)
    return results

def generate_initial_answer (prompt:list, echo:bool = False, q_a:bool = True, model:str = MODEL,
                             max_length:int = MAX_TOKENS, verbose:bool = False, thread_num:int = 1,
                             post_reasoning_prompt:str = '', temp=TEMP, system_prompt=''):
    """
    Generate direct answers to the questions in prompt.
    Call batch_inputs on prompt (i.e. if prompt is a list of multiple parallel questions).

    Args:
    - prompt: list of questions/prompts for the model

    Keyword Args:
    - echo: whether prompt is repeated in response text
    - q_a: whether Q&A format is added to prompt
    - model: name of the model in OpenAI API
    - max_length: max number of tokens in text completion
    - verbose: whether the process prints to the console
    - thread_num: number of threads to use for text generation (maximum parallelism with 1 thread per item in prompt)
    - post_reasoning_prompt: prompt text for after reasoning (i.e. for text repetition task)

    Returns: text generation(s), API response(s), full text of query(ies), logprobability(ies)
    """
    if q_a:
        q_a = ["Q: "+p+"\nA:" if p else p for p in prompt] # pass falsy (i.e. empty string, None) values through
    else:
        q_a = prompt

    batches = batch_inputs(q_a, model, verbose=verbose, thread_num=thread_num)
    if thread_num > 1:
        results = {}
        query_model(batches, model, echo, max_length = max_length, thread_num=thread_num, results=results,
                    post_reasoning_prompt = post_reasoning_prompt, temp=temp, system_prompt=system_prompt)
        # Maintaining results through threading requires mutating the results dict.
    else:
        results = query_model(batches, model, echo, max_length = max_length, thread_num=thread_num,
                              post_reasoning_prompt = post_reasoning_prompt, temp=temp, system_prompt=system_prompt)
    # sort results to match input order
    results = [results[q] if(q and (q in results)) else (None, None, None) for q in q_a]
    response_text, logprobs, response = zip(*results)
    full_queries = [q+' '+r if (q and (r is not None)) else None for q,r in zip(q_a, response_text)]
    # if input or output is invalid, full_prompt is None.
    return (response_text, response, full_queries, logprobs)

def generate (prompt:[list,str], style:str = '', model:str = MODEL, max_length:int = MAX_TOKENS,
              verbose:bool = False, thread_num:int = 1, post_reasoning_prompt:str = '', task='', temp=TEMP):
    """
    Generate a response to the prompt, generating additional prompting based on provided style.

    Args:
    - prompt: single question for the model or a list of such

    Keyword Args:
    - style: style to generate prompt format
    - model: name of the model in OpenAI API
    - max_length: max number of tokens in text completion
    - verbose: whether the process prints to the console
    - thread_num: number of threads to use for text generation (maximum parallelism with 1 thread per item in prompt)
    - post_reasoning_prompt: prompt text for after reasoning (i.e. for text repetition task)
    - task: name of the task (for answer extraction prompt)

    Returns: generated text(s) from the prompt(s), logprobability(ies) of generated text(s),
        full text of prompt(s) after style is applied
    """
    if isinstance(prompt, str):
        prompt = [prompt] # convert single input to list, to be compatible with batch processing

    system_prompt = ''
    if style and ('system:' in style):
        style, system_prompt = style.split('system:')
        style = style.strip()
        system_prompt = system_prompt.strip()
        if verbose:
            print('System Prompt:', system_prompt)

    prompt = [generate_prompt(p, style) for p in prompt]
    if (not style) or ('fewshot' in style or 'direct' in style):
        final_prompt = prompt # for fewshot and direct styles, don't need to add answer extraction to the end.
    else:
        answer_extraction = GSM8K_SUFFIX
        reasoning = generate_initial_answer(prompt, echo=False, q_a=False, model=model,
                                            max_length = max_length, verbose=verbose, thread_num=thread_num,
                                            post_reasoning_prompt = post_reasoning_prompt, temp=temp, system_prompt=system_prompt)
        final_prompt = [r + answer_extraction if r else r for r in reasoning[2]]
        # pass falsy (i.e. empty string, None) values through

    if len(final_prompt) < 1 or all([p is None for p in final_prompt]):
        final = reasoning # if all prompts empty, do not try to continue querying
    else:
        final = generate_initial_answer(final_prompt, echo=False, q_a=False, model=model,
                                        max_length = max_length, verbose=verbose, thread_num=thread_num,
                                        post_reasoning_prompt = post_reasoning_prompt, temp=temp, system_prompt=system_prompt)
    response_text = final[0]
    logprobs = final[3]
    return  response_text, logprobs, final_prompt

def get_cond_logprobs (prompt:str, target:str, prior_prompt:str, style:str = "", model:str = MODEL, verbose:bool = True):
    """
    Get the total per-token logprobability of a target answer given the prompt.

    Args:
    - prompt: single question for the model
    - target: target answer to the prompt (over which logprobabilities are evaluated)
    - prior_prompt: if provided, use that as the prompt.
        - Otherwise, generate the prompt (e.g. chain of thought), using the specified style.

    Keyword Args:
    - style: style to generate prompt format (if not prior_prompt)
    - model: name of the model in OpenAI API
    - verbose: whether the process prints to the console

    Returns: total logprobability of target, tokens included in the logprobability calculation,
        full text of prompt for the target (after style is applied)
    """
    repeated = (isinstance(prior_prompt, (list,str)) and len(prior_prompt) > 0)
    if verbose:
            print("Question:", prompt)
            print("Target completion:", target)
            print('Generating explanations')
            print(f'Repeat answers? {repeated}{", "+str([ans[0] for ans in prior_prompt]) if prior_prompt else ""}')
    if ('sbs' in style): # if step-by-step logic has already been computed, don't re-compute.
        explanation = prior_prompt
    else:
        explanation = generate_prompt(prompt, style) # if explanation is not generated, just use prompt.
        response = generate_initial_answer(explanation, q_a=False, model=model)
        explanation = response[2] + "\nThe answer is:"

    prompt = explanation + ' ' + target
    # adds a space between "The correct answer is:" and target answer.

    if verbose:
        print('Whole prompt:')
        print(explanation)

    time.sleep(WAIT)
    response = openai.Completion.create(prompt=prompt,
    model=model,
    logprobs = 1,
    temperature=0,
    echo=True,
    max_tokens=0, # don't add any more completion on. just calculate logprobs.
    request_timeout=TIMEOUT
    )

    if verbose:
        print('identifying log probabilities')
    tokens = response['choices'][0]['logprobs']['tokens']
    logprobs = response['choices'][0]['logprobs']['token_logprobs']

    ans_logprobs = []
    ans_tokens = []
    answer_found = False
    for i in range(len(tokens)):
        if (tokens[i] == "<|endoftext|>") or (target in ''.join(ans_tokens)): # end of text or entire target discovered
            break
        # check if the last 5 tokens are the end of the explanation text (next token starts the answer)
        elif not answer_found:
            last_few_tokens = ''.join(tokens[max(0,i-4):i+1]).strip()
            answer_found = (last_few_tokens == explanation.strip()[-1*len(last_few_tokens):])
        elif len(ans_tokens) < 1 and len(tokens[i].strip()) == 0: # don't include leading whitespace
            continue
        elif tokens[i].strip() in target: # token match
            ans_tokens.append(tokens[i])
            ans_logprobs.append(logprobs[i])
        else: # completion not in the target - skip
            continue

    if verbose:
        print('Target tokens:', ans_tokens)
        print('Target logprobs:', ans_logprobs)
        print('Target total logprob:', sum(ans_logprobs))
    return (sum(ans_logprobs), ["".join(ans_tokens)], explanation)
