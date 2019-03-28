import argparse
import torch
from laughter_prediction.feature_extractors import LibrosaExtractor


def predicted_to_intervals(pred_classes, frame_sec):
    """
    Extracts target class intervals from binary predictions by frames

    :param pred_classes: array-like, binary prediction for each timeframe
    :param frame_sec: int, length of each timeframe in seconds
    :return: array of pairs, valid target class intervals
    """
    intervals = []
    i = 0
    while i < len(pred_classes):
        if pred_classes[i] == 0:
            i += 1
            continue
        j = i + 1
        while j < len(pred_classes) and pred_classes[j] == 0:
            j += 1
        intervals.append((frame_sec*i, frame_sec*j))
        i = j
    return intervals


def main():
    parser = argparse.ArgumentParser(description='Script for prediction laughter intervals for .wav file')
    parser.add_argument('--wav_path', type=str, help='Path to .wav file')
    parser.add_argument('--predictor_dump', type=str, default='predictor.pkl',
                        help='Path to dumped predictor object')
    args = parser.parse_args()

    predictor = torch.load(args.predictor_dump)
    extractor = LibrosaExtractor(frame_sec=0.5)

    feature_df = extractor.extract_features(args.wav_path)
    pred_classes = predictor.predict(feature_df.as_matrix())
    intervals = predicted_to_intervals(pred_classes, frame_sec=0.5)
    print("Target intervals")
    print(intervals)


if __name__ == '__main__':
    main()
