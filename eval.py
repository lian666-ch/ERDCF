import os
import numpy as np
import data_io

def cal_prf_metrics(pred_list, gt_list, thresh_step=0.001):
    final_accuracy_all = []

    for thresh in np.arange(0.0, 1, thresh_step):
        print(thresh)
        statistics = []

        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))

        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        tn = np.sum([v[3] for v in statistics])

        # calculate precision
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        # calculate recall
        r_acc = tp / (tp + fn)
        iou_t =  tp/(tp+fp+fn)
        iou_b = tn/(tn+fp+fn)
        miou = (iou_b+iou_t)/2
        # calculate f-score
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc),miou])
        print(2 * p_acc * r_acc / (p_acc + r_acc), miou)

    return final_accuracy_all

def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    return [tp, fp, fn,tn]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--metric_mode', type=str, default='prf', help='[prf | sem]')
parser.add_argument('--results_dir', type=str, default='./predict/ERDCF/crack/')
parser.add_argument('--gt_dir', type=str, default='./predict/ERDCF/gt/')
parser.add_argument('--output', type=str, default='./predict/ERDCF/result.crf')
parser.add_argument('--thresh_step', type=float, default=0.01)
args = parser.parse_args()

if __name__ == '__main__':
    metric_mode = args.metric_mode
    results_dir = os.path.join(args.results_dir)
    src_img_list, tgt_img_list = data_io.get_image_pairs(results_dir,args.gt_dir)
    final_results = []

    if metric_mode == 'prf':
        final_results = cal_prf_metrics(src_img_list, tgt_img_list, args.thresh_step)
    else:
        print("Unknown mode of metrics.")
    data_io.save_results(final_results, args.output)