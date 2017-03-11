from __future__ import division
import json
import string
from collections import defaultdict
from collections import Counter


class BaseEvaluator(object):
    def __init__(self, data_path):
        self.dataset = None
        self.data_path = data_path

    def load_dataset(self, datafile):
        with open(datafile, 'r') as f:
            self.dataset = json.load(f)


class Evaluator(BaseEvaluator):
    def __init__(self, data_path):
        super(Evaluator, self).__init__(data_path)
        self.dataset = None
        self.sub_scores = ['overall', 'text', 'diagram', 'overall_text']

    def evaluate_submission(self, predicted_answers):
        if isinstance(predicted_answers, str):
            with open(predicted_answers, 'r') as f:
                predicted_answers = json.load(f)
        if not self.dataset:
            self.load_dataset(self.data_path)
        assert not self.validate_answer_format(predicted_answers)
        n_correct = self.count_correct(predicted_answers)
        n_total = Counter([qid.split('_')[0] for qid in self.dataset])
        n_total['overall'] = sum(n_total.values())
        overall_accuracy = n_correct['overall'] / n_total['overall']
        dq_accuracy = n_correct['DQ'] / n_total['DQ']
        ndq_accuracy = n_correct['NDQ'] / n_total['NDQ']
        scores = {k: v for k, v in zip(self.sub_scores, [overall_accuracy, ndq_accuracy, dq_accuracy, ndq_accuracy])}
        return scores

    def count_correct(self, predicted_answers):
        correct = []
        for qid, ans in predicted_answers.items():
            if ans == self.dataset[qid]:
                correct.append(qid.split('_')[0])
        correct_by_type = Counter(correct)
        correct_by_type['overall'] = sum(correct_by_type.values())
        return correct_by_type

    def answered_correctly(self, gt_answers, q_id, submission_answer):
        return submission_answer == gt_answers[q_id]

    def validate_answer_format(self, predicted_answers):
        errors = defaultdict(list)
        for qid, answer in predicted_answers.items():
            id_prefix, id_number = qid.split('_')
            if id_prefix not in ['DQ', 'NDQ']:
                errors[qid].append('bad id prefix')
            if len(id_number) != 6:
                errors[qid].append('bad id number')
            if answer not in string.ascii_letters:
                errors[qid].append('answer not a letter index')
        all_dataset_questions = self.dataset
        questions_missing = set(all_dataset_questions.keys()).difference(set(predicted_answers.keys()))
        if questions_missing:
            print('***Warning***')
            print('unanswered questions detected')
            print('recording missing files in missing_qids.txt')
            with open('missing_qids.txt', 'w') as f:
                f.write("\n".join(questions_missing))
        if not errors and not questions_missing:
            print('All validation test pass')
        elif not questions_missing:
            print('errors found ')
            for qid, error in errors.items():
                print(qid, ' ', error)
        return errors


