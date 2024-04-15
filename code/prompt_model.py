"""
Implementation of big-bench's abstract model class, to allow the tasks to interact with OpenAI's API.

Class organization loosely inspired by BIG-Bench's Dummy Model class.
(https://github.com/google/BIG-bench/blob/main/bigbench/models/dummy_model.py).
"""
from collections import defaultdict
import os
import time

from api_call import get_cond_logprobs, generate, MAX_TOKENS

class PromptModel():
    def __init__(self,
                 style:str,
                 model:str = 'text-davinci-002',
                 filepath:str = os.path.abspath(os.path.join('..','data','responses_prompt.csv')),
                 verbose:bool = False,
                 thread_num:int = 1,
                 temp:float=0):
        """
        Constructor for PromptModel class.

        Args:
        - style: indicates what prompting style is used (e.g. chain of thought)

        Keyword Args:
        - model: name of the model in OpenAI API (for identifying rate limits)
        - filepath: path to file where responses should be logged (csv)
        - verbose: whether the process prints to the console
        - thread_num: number of threads to use for parallel querying
        """
        self.style = style
        self.model = model
        self.name = model
        self.logprobs = defaultdict(dict)
        self.prev_answers = {}
        self.verbose = verbose
        self.thread_num = thread_num
        self.temp = temp
        self.prompt_style_record = self.style # how to record the prompt style in the csv
        if self.temp:
            self.prompt_style_record += ' temp='+str(self.temp)
            self.prompt_style_record = self.prompt_style_record.replace('"','""') # clean for saving to csv

        try:
            f = open(filepath, 'r')
            f.close()
        except:
            os.makedirs(os.path.split(filepath)[0], exist_ok=True)
            f = open(filepath, 'w') # if it doesn't exist, create file with headers
            f.write('Date/Time,Question,Target Answer,Model Name,Prompt Style,Full Prompt,Answer,Logprob,End Time\n')
            f.close()
        self.f = open(filepath, 'a', encoding="utf-8")

    def generate_text(self, inputs:[str,list], max_length:int = MAX_TOKENS, target:[str,list] = '',
                      post_reasoning_prompt:str = '', task:str = ''):
        """
        Generate text response to inputs (string or list of strings).
        Record log-probabilities to self.logprobs for quick retrieval.

        Args:
        - inputs: single input for the model or a list of such inputs

        Keyword Args:
        - max_length: max number of tokens in text completion
        - target: target response(s) (for record-keeping, does not affect generation)
        - post_reasoning_prompt: prompt text for after reasoning (i.e. for text repetition task)
        - task: name of the task (for answer extraction prompt)

        Returns: response(s) to inputs. If inputs is a string, returns a string.
            - Otherwise returns a list of the same length as inputs.
        """
        if isinstance(inputs, str):
            # inputs is a single string
            start = str(time.asctime(time.localtime()))
            response = generate(inputs,
                                style = self.style,
                                model = self.model,
                                max_length = max_length,
                                verbose = self.verbose,
                                thread_num=self.thread_num,
                                post_reasoning_prompt=post_reasoning_prompt,
                                task = task,
                                temp=self.temp)
            end = str(time.asctime(time.localtime()))
            response_text = response[0][0]
            logprob = response[1][0]
            final_prompt = response[2][0]
            self.logprobs[input][response_text] = logprob

            clean_input = inputs.replace('"','""') # format double quotes for csv
            clean_response = str(response_text).replace('"','""')
            clean_target = str(target).replace('"','""') if target else "<generation>"
            clean_prompt = final_prompt.replace('"','""')
            self.f.write(f'{start},"{clean_input}","{clean_target}",{self.model},"'+
                         f'{self.prompt_style_record}","{clean_prompt}","'+
                         f'{clean_response}",{str(logprob)},{end},\n')

        elif isinstance(inputs, list):
            # inputs is a list
            response_text = []
            start = str(time.asctime(time.localtime()))
            response_text, logprobs, prompts = generate(inputs,
                                                        style = self.style,
                                                        model = self.model,
                                                        max_length = max_length,
                                                        verbose = self.verbose,
                                                        thread_num=self.thread_num,
                                                        post_reasoning_prompt=post_reasoning_prompt,
                                                        task = task,
                                                        temp=self.temp)
            end = str(time.asctime(time.localtime()))
            for i in range(len(response_text)):
                if response_text[i]: # do not record None or empty string responses
                    clean_input = inputs[i].replace('"','""')
                    clean_response = str(response_text[i]).replace('"','""')
                    clean_target = str(target[i]).replace('"','""')
                    clean_prompt = prompts[i].replace('"','""')
                    self.logprobs[inputs[i]][str(response_text[i])] = logprobs[i]
                    self.f.write(f'{start},"{clean_input}","{clean_target}",{self.model},'+
                                 f'"{self.prompt_style_record}","{clean_prompt}","'+
                                 f'{clean_response}",{str(logprobs[i])},{end}\n')

        else:
            raise ValueError("inputs has unexpected type %s" % type(inputs))
        return response_text

    def cond_log_prob(self, inputs:[str,list], targets:list):
        """
        Get the total log probability of targets, conditioned over inputs.
        Log probabilities are not currently supported by the OpenAI chat models.

        Args:
        - inputs: single input for the model or a list of such inputs
        - targets: list of targets to check for each input.
            - If inputs is a single string, targets should be a list of strings.
            - If inputs is a list of n strings, targets should contain n lists (of strings).

        Returns: the total logrobability of each target, conditioned over the corresponding input.
            - Output is a list of the same shape as targets.
        """

        logprobs_out = []
        if isinstance(inputs, str):
            input = inputs
            for target in targets:
                self.f.write(str(time.asctime(time.localtime()))+",")
                self.f.write('"'+input.replace('"',"'")+'","'+target+'",'
                             +self.model+','+self.prompt_style_record+',')
                if target not in self.logprobs[input]:
                    repeat_answers = (self.prev_answers[input] if input in self.prev_answers else False)
                    response = get_cond_logprobs(input,
                                                target,
                                                repeat_answers,
                                                style = self.style,
                                                model = self.model,
                                                verbose = self.verbose)
                    self.logprobs[input][target] = response[0]
                    self.prev_answers[input] = response[2]
                logprobs_out.append(self.logprobs[input][target])
                self.f.write('<logprob calculation>,'+str(self.logprobs[input][target])+',\n')
        elif isinstance(inputs, list):
            # inputs is a list
            logprobs_out = []
            for i in range(len(inputs)):
                input = inputs[i]
                input_probs = []
                for target in targets[i]:
                    self.f.write(str(time.asctime(time.localtime()))+",")
                    self.f.write('"'+input.replace('"',"'")+'","'+target+'",'
                                 +self.model+','+self.prompt_style_record+',')
                    if target not in self.logprobs[input]:
                        repeat_answers = (self.prev_answers[input] if input in self.prev_answers else False)
                        response = get_cond_logprobs(input,
                                                     target,
                                                     repeat_answers,
                                                     style = self.style,
                                                     model = self.model,
                                                     verbose = self.verbose)
                        self.logprobs[input][target] = response[0]
                        self.prev_answers[input] = response[2]
                    input_probs.append(self.logprobs[input][target])
                    self.f.write('<logprob calculation>,'+str(self.logprobs[input][target])+',\n')
                logprobs_out.append(input_probs)
        else:
            raise ValueError("inputs has unexpected type %s" % type(inputs))
        return logprobs_out

    def close_log (self):
        """
        If there is an error, add a newline and close the results file.
        """
        if self.f.closed: # if file is already closed, just return.
            return
        self.f.write('\n')
        self.f.close()
