#!/usr/bin/env python
import sys
import os.path
from evaluate_charades import LocalizationEvaluator
from evaluate_charades import ClassificationEvaluator


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
    submission_path_loc = os.path.join(input_dir, 'res', 'charades_loc_sampl.txt')
    submission_path_class = os.path.join(input_dir, 'res', 'charades_class_sampl.txt')
    data_path = os.path.join(input_dir, 'ref', 'Charades_v1_test.csv')
    submission_scores = {}
    submission_scores['temporal_segmentation'] = evaluate_submission(data_path, submission_path_loc, LocalizationEvaluator)
    submission_scores['action_recognition'] = evaluate_submission(data_path, submission_path_class, ClassificationEvaluator)
    write_score_file(submission_scores, output_dir)


if __name__ == '__main__':
    main()