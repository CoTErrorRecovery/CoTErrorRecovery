from constants import *
from prompt_model import PromptModel

# evaluation scripts, for all datasets
def evaluate_question (question:str, target:str, model:PromptModel = None, response:str = '',
                       scorer:Callable[[str, str], bool] = number_scorer, taskset:str = 'test',
                       task='awps'):
    """
    Evaluate a model on a single question.

    Args:
    - question: the question to be answered
    - target: the target answer

    Keyword Args:
    - model: the model object
    - responsE:
prior generated text. If provided, just pass directly to the scorer without querying the model
    - scorer: the function to score the response. Should return a boolean value corresponding to the score
    - taskset: the task suffix within the dataset (i.e. for the file names)

    Returns: Whether the question matches the target, according to the scorer. (bool)
    """
    if not response: # only query the model if the response is not provided
        response = model.generate_text(question, target=target, task=task+' '+taskset)
    return scorer(response, target), response

def evaluate_batch (questions:list, targets:list, model:PromptModel = None, responses:list = [],
                    scorer:Callable[[str, str], bool] = number_scorer, taskset:str = 'test', limit:int = -1, task='awps'):
    """
    Evaluate a batch of questions.

    Args:
    - questions: the questions to be answered
    - targets: the target answers
    - model: the model object

    Keyword Args:
    - responses: prior generated texts. If provided, just pass directly to the scorer without querying the model
    - scorer: the function to score the response. Should return a boolean value corresponding to the score
    - taskset: the task suffix within the dataset (i.e. for the file names)
    - limit: the maximum number of <prompt, response> pairs to score
        - -1 corresponds to no limit

    Returns: a list of scores between the responses and the targets.
    """
    if not responses: # only query the model if the response is not provided
        responses = model.generate_text(questions, target=targets, task=task+' '+taskset)
    end = time.asctime(time.localtime()) # record end time of model processing

    failures = []
    scores = []
    for q, r,t in zip(questions, responses, targets):
        if len(scores) == limit: # only score specified number of responses
            break
        if not r:  # do not score None/blank responses
            failures.append((q,t))
            continue
        scores.append(scorer(r,t))
    return scores, responses, end, failures

def evaluate_results (inpath:str, outpath:str, skip:int = 0, limit:int = -1,
                      scorer:Callable[[str, str], bool] = number_scorer,
                      task='awps', taskset:str = DEFAULT_TASKSETS['awps'], skip_condition:Callable[[str],bool] = None):
    """
    Score prior results stored in a csv file, using the format generated automatically by the Prompt Model.

    Args:
    - inpath: file path to the previous results file (i.e. from running evaluate_<dataset>.py)
        - Must be a csv containing the text to repeat in the "Question" column
    - outpath: file path to the results summary file

    Keyword Args:
    - skip: number of (non-header) rows to skip from inpath
    - limit: the maximum number of <prompt, response> pairs to score
        - -1 corresponds to no limit
    - scorer: the function to score the response. Should return a boolean value corresponding to the score
    - task: the general dataset name
    - taskset: the task suffix within the dataset(i.e. for the file names)
    - skip_condition: if provided, skip rows for which skip_condition(question) is True.
        - Applied before integer skip argument

    Returns: a dictionary summarizing the results (also saved to outpath)
    """
    if VERBOSE:
        print(f"evaluating previous results from {os.path.split(inpath)[-1]}")
    results_log = []
    summary = []
    questions_seen = 0
    questions = pd.read_csv(inpath, encoding='utf-8', skiprows=(list(range(1,skip)) if skip > 0 else 0),
                            header=0, index_col=False)
    if skip_condition is not None:
        questions = questions[~questions.apply(skip_condition, axis=1)]
    if len(questions) < 1:
        questions = pd.read_csv(inpath.replace("_annotated",""), encoding='utf-8', skiprows=(list(range(1,skip)) if skip > 0 else 0),
                                header=0, index_col=False)
        questions[~questions.apply(skip_condition, axis=1)]
    if VERBOSE:
        print('length of pre_results:', len(questions))
    prev_model = ''
    start = ''
    end = ''
    for i, question in questions.iterrows():
        if skip_condition is not None and skip_condition(question):
            continue # skip this question
        if len(results_log) == 0:
            start = question['Date/Time']
        elif 'End' in question.index:
            end = question['End']
        else:
            end = question['Date/Time']
        model_family = question['Model Name']
        if model_family != prev_model: # if this response is a new model, record previous model results before continuing
            if prev_model and len(results_log)>0: # don't record empty results
                summary.append({'Task':taskset, 'Model Family':prev_model,
                                'Trials': len(results_log),
                                'Scorer': scorer.__name__,
                                'Score': sum(results_log)/len(results_log)})
            results_log=[]
            prev_model=model_family

        q = question['Question']
        response = question['Answer']
        a = question['Target Answer']
        if isinstance(a, str) and '####' in a:
            a = a.split('####')[-1].strip() # extract answer from multistep answer

        score = evaluate_question(q,a,response=response, scorer=scorer, task=task)[0]

        if VERBOSE:
            print('Target:',a)
            print('Response:', response)
            print("Match score:", score)
        results_log.append(score)

        questions_seen += 1
        if questions_seen == limit: # only return the specified number of examples
            break

    if len(results_log) > 0: # record the final model results
        summary.append({'Taskset':taskset, 'Model Family':prev_model, 'Model Name':prev_model,
                        'Trials': len(results_log),
                        'Scorer': scorer.__name__,
                        'Score': sum(results_log)/len(results_log),
                        "Start_Time":start,
                        "End_Time":end})
        prev_results = pd.DataFrame(summary)

    prev_results.to_csv(outpath, encoding='utf-8', index=False, mode='a')
    if VERBOSE:
        print(prev_results.tail())

    return summary

def create_skip_condition (taskset:str, model_name:str, style:str):
    def skip_row (row):
        if isinstance(row['Model Name'], float) or isinstance(row['Prompt Style'], float): # skip blank rows
            return True
        error_type = ERROR_TYPE[taskset.split('position-')[-1].split('_')[0]]
        if 'Model Name' in row and row['Model Name']!=model_name: # incorrect model
            return True
        if 'Prompt Style' in row and row['Prompt Style']!=style: # incorrect prompt style
            return True
        if 'Error Type' in row and isinstance(row['Error Type'], str) and row['Error Type']!=error_type: # incorrect error type
            return True
        return False # passed all tests
    return skip_row
