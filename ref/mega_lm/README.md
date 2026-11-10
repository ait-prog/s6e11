# S6E11 target-first language model, own predictions

GPT-2 small, trained by me on one laptop GPU with `lm_member.py` (this folder), no public predictions used.
Each row is written label first: one learned token per column, then the value's GPT-2 tokens, the columns in a fresh random order.
A row is scored under both labels and Bayes turns the two scores into log-odds.
The target-first idea is Chris Deotte's (S6E10 1st place), with settings from masayakawamata's public S6E11 write-up; the compact row format is mine.

* `lm_oof.npy`: train.csv order, NaN for folds not finished yet. Folds: StratifiedKFold(5, shuffle=True, random_state=42).
* `lm_test.npy`: test.csv order. Only the order of the values matters.

Finished so far: folds [0, 1, 2, 3, 4], 5 fold model(s) on the test rows.
