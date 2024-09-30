import random

def get_random_questions(q):
  N_00 = random.randint(1, 10)
  N_01 = random.randint(1, 10)
  N_02 = random.randint(1, 10)
  N_03 = random.randint(1, 10)
  N_04 = random.randint(1, 10)
  N_05 = random.randint(1, 10)
  N_06 = random.randint(1, 10)
  N_07 = random.randint(1, 10)
  N_08 = random.randint(1, 10)
  N_09 = random.randint(1, 10)
  N_10 = random.randint(1, 10)
  N_11 = random.randint(1, 10)

  formatted_question = q.format(N_00=N_00,
                                N_01=N_01,
                                N_02=N_02,
                                N_03=N_03,
                                N_04=N_04,
                                N_05=N_05,
                                N_06=N_06,
                                N_07=N_07,
                                N_08=N_08,
                                N_09=N_09,
                                N_10=N_10,
                                N_11=N_11)

  return formatted_question


q = "James writes a {N_00}-page letter to {N_01} different friends twice a week.  How many pages does he write a year?"
get_random_questions(q)