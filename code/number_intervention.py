"""
Automate number intervention of chain of thought reasoning.
Inspired by https://www.lesswrong.com/posts/FX5JmftqL2j6K8dn4/shapley-value-attribution-in-chain-of-thought.
"""
import random
from string import ascii_lowercase

from constants import *

VERBOSE=True
MAX_STEPS = 10000 # maximum steps to check for in reasoning


def perturb_value (to_adjust:re.Match, perturbation:str = "random", min_adjust:[int,float] = -3,
                   max_adjust:[int,float] = 3, handle_padding:bool = True):
    """
    Perform the specified perturbation(s) on the given value

    Args:
    - to_adjust (regex match item): text value to adjust

    Keyword Args:
    - perturbation: describes the perturbation style being used. To specify multiple perturbations with matched errors,
        separate each perturbation with "_"
    - min_adjust: minimum adjustment to value
    - max_adjust: maximum adjustment to value
    - handle_padding: whether the adjusted value should be left-padded to the same string length as to_adjust

    Returns: The adjusted value, as a string
    """
    to_adjust = to_adjust.group()
    to_adjust_float = float(to_adjust.replace(',',''))
    number_format = lambda digs: format(digs, ',d') if (',' in to_adjust and not isinstance(digs, str)) else str(digs)
    if VERBOSE:
        print(f"Identified {to_adjust} for adjustment")

    digits = to_adjust.replace(',','').strip().strip('.') # remove commas, trailing whitespace, and trailing period
    if handle_padding and digits[0] == '0' and len(digits) > 1:
        padding = len(to_adjust)
    else:
        padding = 0
    adjusteds = [to_adjust_float]
    while any([float(adjusted) == to_adjust_float for adjusted in adjusteds]):
        adjusteds = []
        for perturb in perturbation.split('_'): # keep track of matched perturbations
            adjusted = digits
            adjusted = eval(adjusted.lstrip('0'))
            if 'random' in perturb: # random integer adjustment within [min_adjust, max_adjust]
                adjustment_amount = random.randint(min_adjust,max_adjust)
                adjusted = adjusted+adjustment_amount
            if 'add' in perturb:
                adjustment_amount = eval(re.match(r'\d+', perturb[perturb.find('add')+len('add'):]).group())
                adjusted = adjusted+adjustment_amount
            adjusteds.append(adjusted)

        after_decimal = len(digits.split('.')[-1]) if '.' in digits else 0
        # round to same number of digits after the decimal as in original number
        adjusteds = [round(adjusted, after_decimal) for adjusted in adjusteds]
        if VERBOSE: print('Adjusted to:',adjusteds)

    adjusted_value = [number_format(adjusted) for adjusted in adjusteds]

    if handle_padding:
        if to_adjust[-1] == '.': # add back trailing period
            adjusted_value = [str(adjusted)+'.' for adjusted in adjusted_value]
        adjusted_value = [str(adjusted).rjust(padding, '0') for adjusted in adjusted_value]
    return adjusted_value

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


def intervention (reasoning:str, task:str = 'test',
                  perturbation:str = 'random',
                  min_adjust:[int,float] = -3, max_adjust:[int,float] = 3,
                  position:str = "any",
                  handle_padding:bool = True,
                  remove_steps:bool = True,
                  nonnumeric:str = ''
                  ):
    """
    Perform number perturbation on chain of thought reasoning.

    Args:
    - reasoning: reasoning text to be perturbed

    Keyword Args:
    - task: name of task/taskset to be adjusted
    - perturbation: describes the perturbation style being used. To specify multiple perturbations with matched errors,
        separate each perturbation with "_"
    - min_adjust, max_adjust: define bounds of minimum/maximum value perturbation.
    - position: what positions of numbers can be perturbed (i.e. error type)
        - ("copy" for copying error, "calc" for calculation error, "propcalc" for propagated calculation error,
            or "any" for any position)
    - handle_padding: whether to maintain zero-padding or not when performing perturbation
    - remove_steps: whether to attempt to remove step numbers when selecting values to perturb
    - nonnumeric: additional, non-numeric adjustments to make to the reasoning text

    Returns: the perturbed reasoning text, discarding any reasoning after the (final) error.
    """
    if len(reasoning) < 1:
        return reasoning

    if COT_PROMPT in reasoning: # only make perturbation after "Let's think step by step."
        reasoning_start = reasoning.index(COT_PROMPT) + len(COT_PROMPT)
    else:
        reasoning_start = 0

    reasoning_no_steps = reasoning # remove step numbers (e.g. 1. <step> 2. <step>)
    if remove_steps:
        step_start = reasoning_start
        # check for steps in order (i.e. should help avoid unnecesary removal of numbers at end of a sentence)
        for step in range(1, MAX_STEPS):
            repl = "@"*len(str(step)) # asterisks to match string length of step number
            if re.search(fr'(\s|^){step}\.\s', reasoning_no_steps):
                # find index of replaced step number
                new_step_start = re.search(rf'(\s|^){step}\.\s', reasoning_no_steps).span()[0]
                reasoning_no_steps = reasoning_no_steps[:step_start] + \
                re.sub(rf'(\s|^){step}\.\s', ' '+repl+'. ', reasoning_no_steps[step_start:], count=1)
                # replace first instance of step number
                step_start = new_step_start
            else: # no more step numbers
                break

    reasoning_end = len(reasoning_no_steps)

    # allow commas every 3 digits, allow decimals, ignore negatives
    numericals = list(re.finditer(r'\d+(,\d{3})*(\.\d*)?', reasoning_no_steps))
    occurrences = {}
    if position != 'any':
        for match in numericals:
            group_cleaned = float(match.group().replace(',','')) # remove commas
            if int(group_cleaned) == group_cleaned:
                group_cleaned = int(group_cleaned)
            if group_cleaned in occurrences:
                occurrences[group_cleaned].append(match) # for checking repeated values, ignore commas
            else:
                occurrences[group_cleaned] = [match]

    if position == 'calc':
        numericals = [matches[0] for matches in occurrences.values()
                        if (matches[0].start() >= reasoning_start) # check that first occurence is in reasoning
                        and (matches[0].start() <= reasoning_end)]
    elif position == 'copy':
        numericals = [match for matches in occurrences.values()
                            for match in matches[1:]
                            if len(matches) > 1] # extract repeated occurrences and flatten list
    numericals = [[match] for match in numericals if (match.start() >= reasoning_start)
                                                and (match.start() <= reasoning_end)]
    # only include numbers whose first appearance is after "Let's think step by step."

    if position == 'propcalc':
        numericals = [matches[:2] for matches in occurrences.values()
                      if len(matches) > 1 # check that there are multiple occurences
                      and (matches[0].start() >= reasoning_start) # check that first occurence is in reasoning
                      and (matches[0].start() <= reasoning_end)]

    assert (len(numericals) > 0), f"No numerals fitting search condition (position='{position}') in reasoning"

    new_numerical = True
    while new_numerical: # if the chosen set of values doesn't work, pick a new one
        reasoning_adjusted = ['' for i in range(len(perturbation.split('_')))]
        adjust_set = random.choice(numericals)
        adjusted_value = None
        for to_adjust in adjust_set:
            adjust_index = to_adjust.start()
            reasoning_pre = reasoning[reasoning_start:adjust_index]
            reasoning_adjusted = [r + reasoning_pre for r in reasoning_adjusted]
            reasoning_start = to_adjust.end() # set up in case there is another number to adjust after this

            if adjusted_value is None: # first number to adjust - need to figure out adjustment
                try:
                    adjusted_value = perturb_value(to_adjust, min_adjust=min_adjust, max_adjust=max_adjust,
                                                   perturbation=perturbation, handle_padding=handle_padding)
                    new_numerical = False # if the value can be adjusted to specified perturbation, don't repeat process
                except ValueError as e:
                    print(e)
                    numericals.remove(adjust_set)
                    reasoning_start = (reasoning.index(COT_PROMPT) + len(COT_PROMPT)
                                       if COT_PROMPT in reasoning else 0)
                    break
                # if the value can't be adjusted according to specified perturbation, restart and pick a different one.

            reasoning_adjusted = [reasoning_adjusted[i] + adjusted_value[i] for i in range(len(reasoning_adjusted))]

    # add non-numeric adjustments
    if nonnumeric:
        if 'typo' in nonnumeric: # random typos
            num_typos = eval(nonnumeric.split('typo')[1])
            alpha_only = [c for c in reasoning_adjusted[0] if c.isalpha()]
            num_alpha = len(alpha_only)
            if num_alpha > num_typos:
                pos = random.sample(range(num_alpha), num_typos)
                reasoning_adjusted = [typo(r, pos) for r in reasoning_adjusted]
            else: # if there are fewer than num_typos alphabetic characters
                raise ValueError(f'Not enough alphabetic characters to make typos. (num_typos ={num_typos}, num_alpha={num_alpha})')

    return reasoning_adjusted

def process_task (dataset:str, task_name:str, limit:int = -1, model:str = "gpt-4", style:str = 'sbs',
                  skip_rows:int = 0, sample_size:int = 0, intervention_kwargs:dict = {}):
    """
    Read in recorded chain-of-thought texts (from csv) and write adjusted chain of thought texts to new json file
    (format {question: adjusted chain of thought}).

    Args:
    - dataset: name of the larger dataset to look in
    - task_name: name of the task, should correspond to the csv and/or jsonl and/or json files for this task.

    Keyword Args:
    - limit: number of questions to process (-1 indicates no limit)
    - model: model to consider responses for
    - skip_rows: number of rows to skip from the beginning of the file
    - sample_size: if provided, use rows marked as "Sample (n = {sample_size})" in the csv.
    - intervention_kwargs: keyword arguments which is passed to the intervention function.

    Returns: dictionary of format {question: [adjusted chain of thought, target answer]}
    """
    cleaned = clean_model_name(model)

    # load in questions, original chain of thought responses
    task_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, task_name+'.csv'))
    task_df = pd.read_csv(task_path, encoding='utf-8', index_col=None)
    cot = task_df.iloc[skip_rows:]
    cot = cot[(cot.apply(lambda row: number_scorer(row['Answer'], row['Target Answer']), axis=1))
              # only look at questions with correct answers
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

    # set `handle_padding` (from intervention keywords)
    if VERBOSE: print('Intervention keyword arguments:', intervention_kwargs)
    if 'handle_padding' in intervention_kwargs:
        handle_padding = intervention_kwargs.pop('handle_padding')
    else:
        handle_padding=((task_name != 'test') or ('perturbation' in intervention_kwargs and
                                                  'transpose' in intervention_kwargs['perturbation']))

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
                adjusted_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, task_name+'_'+'adjusted_'+
                                                             intervention_key+cleaned+'.json'))
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
            adjusted_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, task_name+'_'+'adjusted_'+
                                                         intervention_key+cleaned+'.json'))
            with open(adjusted_path, 'r', encoding="utf-8") as outfile_prev:
                    adjusted = json.load(outfile_prev)
        else: # no keyword arguments for intervention
            adjusted_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset,
                                                         task_name+'_'+'adjusted_'+cleaned+'.json'))
            with open(adjusted_path, 'r', encoding="utf-8") as outfile_prev:
                    adjusted = json.load(outfile_prev)
    except FileNotFoundError: # if adjusted output file doesn't exist, there are no prior results.
        adjusted = {}
        pass # the file will be created later.
    limit += len(adjusted)

    # perform adjusting for each uncompleted question
    if VERBOSE: print(f'Skipping {len(adjusted)} examples.')
    for i, row in cot.fillna('').iterrows(): #use empty string instead of np.NaN/None value for blank answers
        try:
            q = row['Question']
            if q in adjusted and adjusted[q] != '':
                continue
            if VERBOSE: print("Evaluating Question:", q)

            reasoning = str(row['Full Prompt'])
            a = str(row['Target Answer'])
            if VERBOSE:
                print('Unadjusted cot:')
                print(reasoning)
                print()

            if reasoning == '':
                adjusted_reasoning = reasoning
            else:
                adjusted_reasoning = intervention(reasoning, task_name, handle_padding=handle_padding,
                                                  **intervention_kwargs)
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
            adjusted_path = os.path.abspath(os.path.join(RESULTS_PATH, dataset, task_name+'_'+'adjusted_'+\
                                                         intervention_key+cleaned+'.json'))
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
    parser.add_argument("--position", type=str, required=True,
                        help='Error type/position. ("copy" for copying error, "calc" for calculation error, "propcalc" for propagated calculation error, or "any" for any position)')
    parser.add_argument("--perturbation", type=str, default="random",
                        help="Type of perturbation applied to the selected values. 'random' is for experiment 1, and 'add1_add101' is for experiment 2.")
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
                        help="Additional keyword arguments for the perturbation. \
Potential arguments include `min_adjust`, `max_adjust`, `handle_padding`, and `remove_steps`\
Arguments should all be specified like `argname=value`.")
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
    KWARGS = {'position': args.position, 'perturbation': args.perturbation} # These are required
    if args.intervention:
        for kwarg in args.intervention: # These are optional
            argname,value = kwarg.split("=") # parse the argument names and values from the command-line strings
            KWARGS[argname] = value

    process_task(DATASET, TASKSET, limit=LIMIT, model=MODEL, style=STYLE, skip_rows=SKIP_ROWS,
                 sample_size=SAMPLE_SIZE, intervention_kwargs=KWARGS)
