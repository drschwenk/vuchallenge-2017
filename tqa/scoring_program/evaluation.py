#!/usr/bin/env python
import sys
import os.path
from evaluate_tqa import Evaluator


def evaluate_submission(data_path, answer_path, evaluator):
    model_evaluator = evaluator(data_path)
    scores = model_evaluator.evaluate_submission(answer_path)
    return scores


def write_score_file(scores, out_path):
    with open(os.path.join(out_path, 'scores.txt'), 'w') as output_file:
        output_form = ['{0}:{1}\n'.format(metric, score) for metric, score in scores.items()]
        output_file.writelines(output_form)


def main():
    _, input_dir, output_dir = sys.argv
    submission_path = os.path.join(input_dir, 'res', 'submission', 'random_submission.json')
    data_path = os.path.join(input_dir, 'ref', 'val_ans.json')
    submission_scores = evaluate_submission(data_path, submission_path, Evaluator)
    write_score_file(submission_scores, output_dir)


if __name__ == '__main__':
    main()