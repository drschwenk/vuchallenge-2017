from __future__ import division
import numpy as np
import pandas as pd


class CharadesEvaluator(object):
    n_classes = 157

    def __init__(self, data_path):
        self.subm_columns = ['frame_id']
        self.data_path = data_path
        self.gt_labels = None
        self.vid_ids = None
        self.gt_array = None
        self.submission = None
        self.submission_array = None

    def load_groundtruth(self):
        gt_labels = pd.read_csv(self.data_path)
        gt_labels['length'] = pd.to_numeric(gt_labels['length'])
        gt_labels['actions'].fillna('', inplace=True)
        self.gt_labels = gt_labels

    def load_submission(self, submission_file):
        loc_submission = pd.read_csv(submission_file, header=None)
        build_proc_sub = loc_submission[0].str.split(' ').values.tolist()
        proc_sub = pd.DataFrame.from_records(build_proc_sub, columns=[self.subm_columns + range(self.n_classes)])
        num_proc_sub = proc_sub.apply(pd.to_numeric, errors='ignore')
        grouped_by_vid = num_proc_sub
        self.submission = grouped_by_vid

    def compute_scores(self):
        m_aps = []
        for oc_i in range(self.n_classes):
            sorted_idxs = np.argsort(- self.submission_array[:, oc_i])
            tp = self.gt_array[:, oc_i][sorted_idxs] == 1
            fp = np.invert(tp)
            npos = tp.sum()
            fpcs = np.cumsum(fp)
            tpcs = np.cumsum(tp)
            prec = tpcs / (fpcs + tpcs)
            avg_prec = 0
            tmp = self.gt_array[:, oc_i][sorted_idxs]  # for faster lookup in the loop
            for i in range(self.submission_array.shape[0]):
                if tmp[i] == 1:
                    avg_prec += prec[i]
            m_aps.append(avg_prec / npos)
        m_aps = np.array(m_aps)
        m_ap = np.mean(m_aps)
        w_ap = (m_aps * self.gt_array.sum(axis=0) / self.gt_array.sum()).sum()
        return m_ap, w_ap

    def evaluate_submission(self, submission_file):
        self.load_groundtruth()
        self.load_submission(submission_file)
        self.build_ground_truth_array()
        self.build_aligned_submission_array()

        mean_ap, weighted_ap = self.compute_scores()
        return {"MAP": mean_ap, "WAP": weighted_ap}


class LocalizationEvaluator(CharadesEvaluator):
    n_classes = CharadesEvaluator.n_classes
    n_frames = 25

    def __init__(self, data_path):
        super(LocalizationEvaluator, self).__init__(data_path)
        self.subm_columns.append('frame_n')

    @staticmethod
    def check_time_range(timepoints, start, stop):
        return start <= timepoints <= stop

    def build_ground_truth_vects(self, act_seq, vid_len, time_checker):
        frame_actions = np.zeros((self.n_frames, self.n_classes))
        time_seq = np.linspace(0, vid_len, self.n_frames, endpoint=False)
        act_seq = act_seq.split(';')
        if not act_seq[0]:
            return frame_actions
        for act in act_seq:
            act_id, start, end = act.split(' ')
            start = float(start)
            end = float(end)
            act_idx = int(act_id[1:])
            activated_time_idxs = time_checker(time_seq, start, end)
            frame_actions[activated_time_idxs, [act_idx]] = 1
        return frame_actions

    def build_ground_truth_array(self):
        check_times_vectorized = np.vectorize(self.check_time_range)
        img_arrays = []
        vid_ids = []
        for i in range(self.gt_labels.shape[0]):
            row = self.gt_labels.iloc[i]
            vid_ids.append(row['id'])
            img_gt_arr = self.build_ground_truth_vects(row['actions'], row['length'], check_times_vectorized)
            img_arrays.append(img_gt_arr)
        comb_gt_array = np.vstack(img_arrays)
        self.vid_ids = vid_ids
        self.gt_array = comb_gt_array

    def build_aligned_submission_array(self):
        aligned_subm_array = np.ones((len(self.vid_ids) * self.n_frames, self.n_classes))
        self.submission = self.submission.groupby('frame_id')
        for gidx, g in enumerate(self.vid_ids):
            subm_img_df = self.submission.get_group(g)
            sort_subm_img_arr = subm_img_df.sort_values('frame_n').values[:, 2:]
            aligned_subm_array[gidx * self.n_frames: (gidx + 1) * self.n_frames:] = sort_subm_img_arr
        self.submission_array = aligned_subm_array


class ClassificationEvaluator(CharadesEvaluator):
    n_classes = CharadesEvaluator.n_classes
    n_frames = 1

    def __init__(self, data_path):
        super(ClassificationEvaluator, self).__init__(data_path)

    def build_ground_truth_vects(self, act_seq):
        frame_actions = np.zeros(self.n_classes)
        act_seq = act_seq.split(';')
        if not act_seq[0]:
            return frame_actions
        for act in act_seq:
            act_id, _, _ = act.split(' ')
            act_idx = int(act_id[1:])
            frame_actions[act_idx] = 1
        return frame_actions

    def build_ground_truth_array(self):
        img_arrays = []
        vid_ids = []
        for i in range(self.gt_labels.shape[0]):
            row = self.gt_labels.iloc[i]
            vid_ids.append(row['id'])
            img_gt_arr = self.build_ground_truth_vects(row['actions'])
            img_arrays.append(img_gt_arr)
        comb_gt_array = np.vstack(img_arrays)
        self.vid_ids = vid_ids
        self.gt_array = comb_gt_array

    def build_aligned_submission_array(self):
        aligned_subm_array = np.ones((len(self.vid_ids) * self.n_frames, self.n_classes))
        for gidx, g in enumerate(self.vid_ids):
            subm_img_arr = self.submission[self.submission['frame_id'] == g].values[:, 1:]
            if not self.gt_labels[self.gt_labels['id'] == g]['actions'].item():
                subm_img_arr = - np.inf * np.ones_like(subm_img_arr)
            aligned_subm_array[gidx * self.n_frames: (gidx + 1) * self.n_frames:] = subm_img_arr
        self.submission_array = aligned_subm_array
