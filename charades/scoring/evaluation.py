#!/usr/bin/env python
import sys
from vuc_task_evaluation.score_submission import score


def main():
    _, input_dir, output_dir = sys.argv
    score(input_dir, output_dir)


if __name__ == '__main__':
    main()
