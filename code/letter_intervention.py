"""
Automate letter intervention of chain of thought reasoning (i.e. adding noise to context).
"""
import random
from string import ascii_lowercase

from constants import *

VERBOSE=True
MAX_STEPS = 10000 # maximum steps to check for in reasoning

def typo (text,positions):
    """
    Add random typos to letters in the text.

    Args:
    - text: the text to typo
    - positions: the positions to adjust, ignoring non-alphabetic characters.
    """
    typoed = ''
    pos = -1
    for c in text:
        if c.isalpha() and pos in positions:
            pos += 1
            typo_c = random.choice(ascii_lowercase)
            print(f"Typoing the {pos+1}th letter, {c}, to {typo_c}.")
            typoed += typo_c
            continue
        elif c.isalpha():
            pos += 1
        typoed += c
    return typoed


def perturbation_num (p):
    match = re.search(r'\d+$', p)  # \d+$ pattern to find digits at the end of the string
    return int(match.group()) if match else None


def intervention (reasoning:str, task:str = 'test',
                  perturbation:str = 'typo1'
                  ):
    """
    Perform letter perturbation on chain of thought reasoning.

    Args:
    - reasoning: reasoning text to be perturbed

    Keyword Args:
    - task: name of task/taskset to be adjusted
    - perturbation: describes the perturbation style being used. To specify multiple perturbations with matched errors,
        separate each perturbation with "_"

    Returns: the perturbed reasoning text, discarding any reasoning after the (final) error.
    """
    if len(reasoning) < 1:
        return reasoning

    if COT_PROMPT in reasoning: # only make perturbation after "Let's think step by step."
        reasoning_start = reasoning.index(COT_PROMPT) + len(COT_PROMPT)
    else:
        reasoning_start = 0

    # add non-numeric adjustments
    perturbations = perturbation.split('_')
    max_num = max([perturbation_num(p) for p in perturbations if perturbation_num(p) is not None])
    alpha_only = [c for c in reasoning[reasoning_start:] if c.isalpha()]
    num_alpha = len(alpha_only)
    if num_alpha > max_num:
        pos_all = random.sample(range(num_alpha), max_num)
    else: # if there are fewer than max_num alphabetic characters
        raise ValueError(f'Not enough alphabetic characters to make typos. (num_typos ={max_num}, num_alpha={num_alpha})')

    reasoning_adjusted = []
    for p in perturbations:
        print(f"Performing perturbation: {p}")
        if 'typo' in p: # random typos
            num_typos = perturbation_num(p)
            pos = pos_all[:num_typos+1]
            reasoning_adjusted.append(typo(reasoning[reasoning_start:], pos))

    return reasoning_adjusted

def process_task (dataset:str, task_name:str, limit:int = -1, model:str = "gpt-4", style:str = 'sbs',
                  skip_rows:int = 0, sample_size:int = 0, intervention_kwargs:dict = {}):
    """
    Read in recorded chain-of-thought texts (from csv) and write adjusted chain of thought texts to new json file
    (format {question: adjusted chain of thought}).

    Args:
    - task_name: name of the task, should correspond to the csv and/or jsonl and/or json files for this task.

    Keyword Args:
    - limit: number of questions to process (-1 indicates no limit)
    - model: model to consider responses for
    - skip_rows: number of rows to skip from the beginning of the file. does not apply to pre-perturbed files.
    - sample_size: if provided, use rows marked as "Sample (n = {sample_size})" in the csv.
    - intervention_kwargs: keyword arguments which is passed to the intervention function.

    Returns: dictionary of format {question: [adjusted chain of thought, target answer]}
    """

    # load in questions, original chain of thought responses
    mode = 'csv' # read from csv by default
    if 'adjusted' in task_name:
        task_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, f"{task_name}_annotated.csv"))
        if os.path.isfile(task_path): # check for annotated version first (to filter error types)
            if VERBOSE: print("loading pre-perturbed, annotated CoT:")
            error_type = ERROR_TYPE[task_path.split('position-')[-1].split('_')[0]]
            cot = pd.read_csv(task_path, encoding='utf-8', index_col=None)
            cot = cot[(cot['Error Type'] == error_type) # only look at questions with correct error type
                & (cot['Model Name'] == model)
                & (cot['Prompt Style'] == style)] # only look at responses from the target model, style
        else: # if no annotated version exists, go directly from non-annotated version
            if VERBOSE: print("loading pre-perturbed, unannotated CoT:")
            mode = 'json'
            task_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, f"{task_name}_{model.replace('.','-')}.json"))
            with open(task_path) as outfile_prev:
                task_dict = json.load(outfile_prev)
    else:
        task_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, task_name+'.csv'))
        task_df = pd.read_csv(task_path, encoding='utf-8', index_col=None)
        cot = task_df.iloc[skip_rows:]
        cot = cot[(cot.apply(lambda row: number_scorer(row['Answer'], row['Target Answer']), axis=1))
                # only look at questions with correct answers in original CoT
                & (cot['Model Name'] == model)
                & (cot['Prompt Style'] == style)] # only look at responses from the target model, style
        assert cot.shape[0] > 0, f"No valid recorded responses for the model: {model}"

        if sample_size: # use sample
            sample_col = f'Sample (n={sample_size}, model={model})'
            if sample_col in cot.columns:
                cot = cot[cot[sample_col]]
            else: # sample of specified size doesn't exist
                if VERBOSE:
                    print(f"Generating sample of size n={sample_size}.")
                cot = cot.sample(n=sample_size)
                task_df[sample_col] = task_df.index.isin(cot.index)
                task_df.to_csv(task_path, index=False) # add row sample markers to csv
        if VERBOSE: print("responses table dimensions:", cot.shape)

    # check for stimuli that have been created previously
    adjusted = {}
    try:
        if 'perturbation' in intervention_kwargs: # perturbations specified
            perturbations = intervention_kwargs['perturbation'].split("_")
            for i in range(len(perturbations)):
                perturbation = perturbations[i]
                temp_kwargs = intervention_kwargs.copy() # copy of intervention keywords for just this perturbation
                temp_kwargs['perturbation'] = perturbation
                intervention_key = '_'.join(map(lambda i: f'{i[0]}-{i[1]}', temp_kwargs.items())) + '_'
                adjusted_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, task_name+'_letter_'+
                                                             intervention_key+model.replace('.','-')+'.json'))
                with open(adjusted_path, 'r', encoding="utf-8") as outfile_prev:
                    prev = json.load(outfile_prev)
                if i == 0: # first previous file - add all to adjusted
                    adjusted = {k:[tuple(v)] for k,v in prev.items()}
                    continue
                for k in adjusted:
                    if k not in prev: # remove keys that aren't in all of the previous files
                        adjusted.pop(k)
                    else: # key in at least one previous file
                        adjusted[k].append(tuple(prev[k]))
        elif len(intervention_kwargs) > 0: # other other intervention keyword arguments specified
            intervention_key = '_'.join(map(lambda i: f'{i[0]}-{i[1]}', intervention_kwargs.items())) + '_'
            adjusted_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, task_name+'_letter_'+
                                                         intervention_key+model.replace('.','-')+'.json'))
            with open(adjusted_path, 'r', encoding="utf-8") as outfile_prev:
                    adjusted = json.load(outfile_prev)
        else: # no keyword arguments for intervention
            adjusted_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset,
                                                         task_name+'_letter_'+model.replace('.','-')+'.json'))
            with open(adjusted_path, 'r', encoding="utf-8") as outfile_prev:
                    adjusted = json.load(outfile_prev)
    except FileNotFoundError: # if adjusted output file doesn't exist, there are no prior results.
        adjusted = {}
        pass # the file will be created later.
    limit += len(adjusted)

    # perform adjusting for each uncompleted question
    if VERBOSE: print(f'Skipping {len(adjusted)} examples.')
    if mode == 'json':
        for q, r in task_dict.items():
            try:
                if q in adjusted and adjusted[q] != '':
                    continue
                reasoning, a = r
                if VERBOSE:
                    print("Evaluating Question:", q)
                    print('Unadjusted cot:')
                    print(reasoning)
                    print()
                if reasoning == '':
                    adjusted_reasoning = reasoning
                else:
                    adjusted_reasoning = intervention(reasoning, task_name, **intervention_kwargs)
                if VERBOSE:
                    print("Adjusted cot:")
                    print(adjusted_reasoning)

                if isinstance(adjusted_reasoning, str): # single perturbation
                    adjusted[q] = (adjusted_reasoning.strip(), a)
                else: # multiple perturbations
                    adjusted[q] = [(r.strip(), a) for r in adjusted_reasoning]
                if len(adjusted) == limit:
                    break
            except Exception as e:
                print(e)
            print('-'*12)
            print()
    else:
        for i, row in cot.fillna('').iterrows(): #use empty string instead of np.NaN/None value for blank answers
            try:
                q = row['Question'].split('A: Let\'s think')[0].strip().split('Q: ')[1]
                if q in adjusted and adjusted[q] != '':
                    continue
                if VERBOSE: print("Evaluating Question:", q)

                if COT_PROMPT in row['Question']: # pre-perturbed cot - perturbed reasoning is in the "Question"
                    reasoning = row['Question'].split(COT_PROMPT)[1]
                else: # original cot - reasoning is in the "Full Prompt"
                    reasoning = str(row['Full Prompt'])
                a = str(row['Target Answer'])
                if VERBOSE:
                    print('Unadjusted cot:')
                    print(reasoning)
                    print()

                if reasoning == '':
                    adjusted_reasoning = reasoning
                else:
                    adjusted_reasoning = intervention(reasoning, task_name, **intervention_kwargs)
                if VERBOSE:
                    print("Adjusted cot:")
                    print(adjusted_reasoning)

                if isinstance(adjusted_reasoning, str): # single perturbation
                    adjusted[q] = (adjusted_reasoning.strip(), a)
                else: # multiple perturbations
                    adjusted[q] = [(r.strip(), a) for r in adjusted_reasoning]
                if len(adjusted) == limit:
                    break
            except Exception as e:
                print(e)
            print('-'*12)
            print()

    # split up multiple perturbations into different files
    if 'perturbation' in intervention_kwargs:
        perturbations = intervention_kwargs['perturbation'].split('_')

        results = {perturbations[i]: {q:r[i] for q,r in adjusted.items()} for i in range(len(perturbations))}
        for perturbation in results:
            intervention_kwargs['perturbation'] = perturbation
            # format intervention keywords for filename
            intervention_key = '_'.join(map(lambda i: f'{i[0]}-{i[1]}', intervention_kwargs.items())) + '_'
            adjusted_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, task_name+'_'+'letter_'+\
                                                         intervention_key+model.replace('.','-')+'.json'))
            with open(adjusted_path, "w", encoding="utf-8") as outfile: # even if there's an error,
                    outfile.write(json.dumps(results[perturbation], indent=4))

    else: # only a single perturbation
        with open(adjusted_path, "w", encoding="utf-8") as outfile: # save results so far, in case there's an error
            outfile.write(json.dumps(adjusted, indent=4))

    return adjusted

if __name__ == "__main__": # run intervention
    # Set up command-line arguments parser
    parser = argparse.ArgumentParser(description="Run experiments with specified conditions.")

    # Add arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model to focus on.")
    parser.add_argument("--style", type=str, default='sbs',
                        help="Prompt Style of the model to focus on.")
    parser.add_argument("--perturbation", type=str, default="typo10",
                        help="Type of perturbation applied to the selected values.")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset to look at.")
    parser.add_argument("--taskset", type=str, default=None,
                        help="Task within the dataset.")
    parser.add_argument("--sample", type=int, default=300,
                        help="Sample size of questions to perturb.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Number of questions to perturb in this run (out of sample). -1 indicates no limit.")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip a certain number of rows from the original chain of thought record.")
    parser.add_argument("--intervention", nargs='*',
                        help="Additional keyword arguments for the perturbation.")
    args = parser.parse_args()

    # set specified perturbation.
    DATASET = args.data
    TASKSET = args.taskset
    if TASKSET is None:
        TASKSET = DEFAULT_TASKSETS[DATASET]
    SAMPLE_SIZE = args.sample
    LIMIT = args.limit
    MODEL = args.model
    STYLE = args.style
    SKIP_ROWS = args.skip

    # Set intervention_kwargs
    KWARGS = {'perturbation': args.perturbation} # These are required
    if args.intervention:
        for kwarg in args.intervention: # These are optional
            argname,value = kwarg.split("=") # parse the argument names and values from the command-line strings
            KWARGS[argname] = value

    process_task(DATASET, TASKSET, limit=LIMIT, model=MODEL, style=STYLE, skip_rows=SKIP_ROWS,
                 sample_size=SAMPLE_SIZE, intervention_kwargs=KWARGS)
