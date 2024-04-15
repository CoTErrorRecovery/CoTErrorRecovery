"""
Read in problems from SVAMP dataset and evaluate a model on them.
Assumes that the SVAMP github repo
    (github.com/arkilpatel/SVAMP/tree/main)
    is cloned in a sister directory to this repository.
"""
from constants import *
from evaluation_utils import *

SVAMP_PATH = os.path.abspath(os.path.join('..','..','SVAMP'))
SVAMP_RESULTS_PATH = os.path.join(RESULTS_PATH,'svamp')

VERBOSE= True

def run_SVAMP (model:PromptModel, skip:int = 0, limit:int = 1, scorer:Callable[[str, str],bool] = number_scorer,
               taskset:str = 'SVAMP', skip_condition:Callable[[str],bool] = None, batch:bool = True,
               questions:list = [], answers:list = []):
    """
    Load SVAMP dataset and evaluate the model over it.
    Always verbose (run with nohup or similar system to ignore console output).

    Args:
    - model: the model object

    Keyword Args:
    - skip: number of rows to skip
    - limit: the maximum number of <prompt, response> pairs to score
        - -1 corresponds to no limit
    - scorer: the function to score the response. Should return a boolean value corresponding to the score
    - taskset: the task name within SVAMP
    - skip_condition: if provided, skip rows for which skip_condition(question) is True
        - Applied before integer skip argument
    - batch: whether the queries to the model should be batched
    - questions: fixed list of questions to be evaluated
        - If provided, ignore the task file
    - answers: fixed list of target answers to the questions list
        - If provided, ignore the task file

    Returns: a dictionary summarizing the results, a list of any failed questions
    """
    model_name = model.name
    model_style = model.style

    built_in = ['SVAMP'] # tasksets which are built into SVAMP

    # if questions kwarg is not provided, read in from task file
    if len(questions) < 1:
        inputs = []
        if taskset in built_in: # built-in tasks from SVAMP are stored in jsonl files
            task_path = os.path.join(SVAMP_PATH, taskset+'.json')
            with open(task_path, 'r', encoding='utf-8') as json_file:
                inputs = json.load(json_file)
            questions = [i["Body"] +' '+ i["Question"] for i in inputs]
            answers = [i['Answer'] for i in inputs]
        else: # custom tasks (from perturbations) are stored in json files
            task_path = os.path.abspath(os.path.join(SVAMP_RESULTS_PATH,
                                                     taskset+'_'+model_name.split(" ")[-1].replace('.','-').replace(':','-')+'.json'))
            with open(task_path, 'r', encoding='utf-8') as json_file:
                inputs = json.load(json_file)
            questions = inputs.keys()
            answers = [a for r,a in inputs.values()]
            reasonings = [r for r,a in inputs.values()]
            # add chain of thought prompt styling to connect question and reasoning
            questions = [generate_prompt(q, model.style)+' '+r if r else q for q,r in zip(questions,reasonings)]

    assert len(questions) == len(answers), "`questions` and `answers` lists are different lengths"

    # match questions and answers into a single list for coordinated skipping
    question_iter = list(zip(questions, answers))
    if skip_condition is not None: # skip all invalid questions
        question_iter = [(q,a) for q,a in question_iter if not skip_condition(q)]
    if limit < 0: # if limit is negative, do all available questions
        limit = len(question_iter) - skip

    if VERBOSE:
        print('Number of valid questions in range:', len(question_iter))
    if len(question_iter) < 1: # if no valid questions, return empty results.
        if VERBOSE:
            print("No valid questions found in range.")
        return {},[]

    question_iter = question_iter[skip:skip+limit] # skip first <skip> valid questions

    # begin evaluation
    start = time.asctime(time.localtime())
    missed_questions = []
    results_log = []
    response_log = {} # tracks the API responses from the model. not essential for this code, but useful for debugging.
    if batch: # batched mode
        questions, answers = tuple(zip(*question_iter)) # separate questions, answers back out to match evaluate_batch
        questions = list(questions)
        answers = list(answers)
        results_log, response_log, end, missed_questions = evaluate_batch(questions, answers, model=model,
                                                                          scorer=scorer, taskset=taskset, task='svamp')
    else: # non-batched mode
        for q, a in question_iter:
            try:
                score, response = evaluate_question(q, a, model=model, scorer=scorer, taskset=taskset, task='svamp')
            except Exception as e: # There was an error in evaluating the question - add to missed_questions log.
                if VERBOSE:
                    print(e)
                missed_questions.append((q,a))
                continue
            if VERBOSE:
                print('Target:',a)
                print("Match score:", score)
            results_log.append(score)
            response_log[q] = response
            if len(results_log) == limit: # only return the specified number of examples
                if VERBOSE:
                    print(f"Max examples ({limit}) reached.")
                break
    end = time.asctime(time.localtime())

    summary = {}
    if len(results_log) > 0: # only record results if at least one question was successful
        summary = {'Taskset':taskset,
                   'Model Name': model_name,
                   'Style': model_style,
                   'Trials': len(results_log),
                   'Scorer': scorer.__name__,
                   'Score': sum(results_log)/len(results_log),
                   "Start_Time":start,
                   "End_Time":end}
    return summary, missed_questions

if __name__ == "__main__": # run evaluation
    # Set up command-line arguments parser
    parser = argparse.ArgumentParser(description="Run experiments with specified conditions.")

    # Add arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model to be evaluated.")
    parser.add_argument("--temp", type=float, default=0,
                        help="Sampling temperature.")
    parser.add_argument("--taskset", type=str, default=DEFAULT_TASKSETS['svamp'],
                        help="Task set to be evaluated (i.e. SVAMP).")
    parser.add_argument("--evaluate", type=str, default='new',
                        help="Whether to query for new answers or just score pre-recorded responses.")
    parser.add_argument("--style", type=str, default='sbs',
                        help="Prompt style. See prompt_model.py and api_call.py for more explanation of the styles.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Number of questions to evaluate. Should be -1 (no limit) if using `--evaluate old`")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip a certain number of questions.")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads to use for the evaluation.")
    parser.add_argument("--scorer", type=str, default='number_scorer',
                        help="Name of the scoring function to use for evaluation. If not number_scorer, must be imported/added to the py file.")
    parser.add_argument("--rerun", type=int, default=1,
                        help="Number of times to re-attempt errored questions.")
    args = parser.parse_args()

    # Set experimental conditions
    EVALUATE = args.evaluate
    MODEL_NAME = args.model
    TEMP = args.temp
    STYLE = args.style
    LIMIT = args.limit
    SKIP = args.skip
    THREADS = args.threads
    TASKSET = args.taskset
    SCORER = eval(args.scorer)
    MAX_RERUN = args.rerun
    prompt_style_record = STYLE
    if TEMP:
        prompt_style_record += ' temp='+str(TEMP)

    # set up filepaths for recording results
    results_path = os.path.abspath(os.path.join(SVAMP_RESULTS_PATH,TASKSET+'.csv'))
    summary_path = os.path.abspath(os.path.join(SVAMP_RESULTS_PATH, 'results_summary.csv'))

    # check for previous evaluations
    pre_results = []
    if os.path.isfile(results_path):
        pre_results = pd.read_csv(results_path, index_col=False, engine='python')
        pre_results = pre_results[pre_results['Model Name'] == MODEL_NAME]
        pre_results = pre_results[pre_results['Prompt Style'] == prompt_style_record]
        if VERBOSE:
            print('length of pre_results:', pre_results.shape[0])
        correct = pre_results.apply(lambda row: SCORER(row['Answer'], row['Target Answer']), axis=1)
        correct = pre_results[correct]
        def skip_condition (q):
            # don't evaluate questions that we've already done.
            if pre_results['Question'].any() and (pre_results['Question'].str.replace('\r','') # get rid of extra carriage returns from excel
                 .str.find(q.replace('\r','').strip()) > -1).any():
                return True
            return False


            # # don't evaluate questions that we haven't already done or have already gotten right.
            # if (pre_results['Question'].str.replace('\r','') # get rid of extra carriage returns from excel
            #      .str.find(q.replace('\r','').strip()) > -1).any():
            #     if (correct['Question'].str.replace('\r','') # skip if correct
            #         .str.find(q.replace('\r','').strip()) > -1).any():
            #         return True
            #     return False # keep if previously done incorrectly
            # return True # skip if new
    else:
        def skip_condition (q): # no prior questions to skip
            return False

    if EVALUATE == 'new': # query the model for new results
        model = PromptModel(STYLE, model=MODEL_NAME, verbose=VERBOSE, filepath=results_path, thread_num=THREADS, temp=TEMP)
        error_results = []
        results_all = []
        rerun = 0

        def cleanup (signum, frame):
            """
            Clean up and close results files.
            """
            if signum:
                print(f'Process killed on {time.asctime(time.localtime())}, with signal number {signum}.')
            model.close_log() # close results file
            if not os.path.exists(summary_path): # if the summary file is new, add the header
                with open(summary_path, 'w') as summary_file:
                    summary_file.write(','.join(results_all[0].keys()).title()+',\n')
            pd.DataFrame(results_all).to_csv(summary_path, mode='a', header=False, index=False, encoding='utf-8') # write to summary file
            sys.exit()
        signal.signal(signal.SIGTERM, cleanup)
        signal.signal(signal.SIGINT, cleanup)

        try:
            # first pass of evaluation
            skip_limit = 7500-len(pre_results) if LIMIT < 0 else SKIP+LIMIT
            skip_interval = 100 if LIMIT > 100 or LIMIT < 0 else LIMIT
            for SKIP in range(SKIP, skip_limit, skip_interval): # temporary thing to help slow batching
                start = time.asctime(time.localtime())
                if VERBOSE:
                    print(f'Evaluating SVAMP {TASKSET} on model: {model.name+" "+model.style}. Start time: {start}')
                task_result_dict = {'Start_Time':start}
                results, errors = run_SVAMP(model, limit=skip_interval, skip=SKIP, taskset=TASKSET,
                                            scorer=SCORER, batch=(THREADS>1), skip_condition=skip_condition)
                task_result_dict.update(results)
                end = time.asctime(time.localtime())
                task_result_dict['End_Time'] = end
                if len(results) > 0 and results['Trials']: # ignore runs with no successful trials
                    results_all.append(task_result_dict)
                error_results.extend(errors)
                if VERBOSE:
                    print(pd.DataFrame(results_all)) # just for verboseness
                    print("Cumulative # of questions with errors:", len(error_results))

            # re-run questions that had errors
            while len(error_results)>0 and rerun<MAX_RERUN:
                time.sleep(60) # allow the rate limit to clear
                rerun += 1
                questions, answers = tuple(zip(*error_results))
                questions = list(questions)
                start = time.asctime(time.localtime())
                if VERBOSE:
                    print(f'Re-evaluating SVAMP {TASKSET} on model: {model.name} for the {rerun}th time. Start time: {start}')
                task_result_dict = {'Start_Time':start}
                results, repeat_errors = run_SVAMP(model, limit=-1, taskset=TASKSET, questions= questions,
                                                   answers= answers, scorer=SCORER, batch=(THREADS>1))
                task_result_dict.update(results)
                end = time.asctime(time.localtime())
                task_result_dict['End_Time'] = end
                if len(results) > 0  and results['Trials']: # ignore runs with no successful trials
                    results_all.append(task_result_dict)
                error_results = repeat_errors
                if VERBOSE:
                    print(pd.DataFrame(results_all)) # just for verboseness
                    print("Cumulative # of questions with errors:", len(error_results))
        except Exception as e:
            if VERBOSE:
                print(e)
        finally: # clean up at the end
            cleanup(None,None)

    else: # scoring pre-recorded results
        anno_path = results_path.replace('.csv','_annotated.csv') # check for annotated version
        if os.path.isfile(anno_path):
            results_path = anno_path
        def not_model (row):
            if isinstance(row['Model Name'], float) or isinstance(row['Prompt Style'], float): # skip blank rows
                return True
            error_type = ERROR_TYPE[TASKSET.split('position-')[-1].split('_')[0]]
            if 'Error Type' in row and row['Error Type']!=error_type: # incorrect error type
                return True
            if 'Model Name' in row and row['Model Name']!=MODEL_NAME: # incorrect model
                return True
            if 'Prompt Style' in row and row['Prompt Style']!=STYLE: # incorrect prompt style
                return True
            return False # passed all tests
        if VERBOSE:
            print(evaluate_results(results_path, summary_path, skip=SKIP, limit=LIMIT,
                                           taskset=TASKSET, scorer=SCORER, skip_condition=not_model))
