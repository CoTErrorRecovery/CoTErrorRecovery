import random
import yaml



def get_new_question_and_answer(data_entry):
  keys = list(data_entry.keys())
  variables = [key for key in keys if len(key) == 4 and "N_" in key]
  formulae = [key for key in keys if len(key) != 4 and "N_" in key]
  variables_with_formulae = [var for var in variables if any(var in formula for formula in formulae)]
  variables_without_formulae = [var for var in variables if var not in variables_with_formulae]

  variable_to_randomvalues = {}
  formulae_to_substitutedvalues = {}

  for var in variables_without_formulae:
    variable_to_randomvalues[var] = random.randint(1, 50)

  for formula in formulae:
    try:
      formula_with_values = data_entry[formula]
      formula_var = formula.replace("_formula", "").strip()
      for var in variable_to_randomvalues:
        formula_with_values = formula_with_values.replace(var, str(variable_to_randomvalues[var]))

      formulae_to_substitutedvalues[formula] = formula_with_values
      final_value = eval(formula_with_values)

      variable_to_randomvalues[formula_var] = final_value

    except Exception as e:
      print(e)
      print(f"error calculating value for {formula}")
    

  combined = {**variable_to_randomvalues, **formulae_to_substitutedvalues}

  new_question = data_entry['question_template'].format(**combined)
  new_answer = data_entry['answer_template'].format(**combined)

  return new_question, new_answer


with open('/content/parameterized_gsm8k.yml', 'r') as file:
    data = yaml.safe_load(file)

for data_entry in data:
  print(get_new_question_and_answer(data_entry))
