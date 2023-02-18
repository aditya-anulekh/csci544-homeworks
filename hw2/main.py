"""
Things to try
TODO: Check the performance of the model by converting everything to lowercase
TODO: Check the performance of the model after removing numbers
"""

import os
import json
from itertools import chain
from tqdm import tqdm
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
TRAIN_DATA = os.path.join(DATA_ROOT, 'train')
VAL_DATA = os.path.join(DATA_ROOT, 'dev')
TEST_DATA = os.path.join(DATA_ROOT, 'test')
DEBUG = False

START_TAG = '<START>'


def read_data(filepath, train_data=True):
    min_freq_thresh = 3
    with open(filepath, 'r') as file:
        lines = file.readlines()

    lines = [line.rstrip('\n') for line in lines]
    lines = [line.split('\t') for line in lines]

    # Creating a vocab file
    counts = {}
    tags = []
    for line in tqdm(lines):
        if len(line) == 3:
            _, word, tag = line
            tags.append(tag)
            if counts.get(word) is not None:
                counts[word] += 1
            else:
                counts[word] = 1

    # Sort words by number of occurrences
    counts = {k: v for k, v in sorted(counts.items(),
                                      key=lambda item: item[1],
                                      reverse=True)}

    words = ['<unk>']

    out_lines = []
    unknown_count = 0

    for i, (word, count) in enumerate(counts.items(), start=1):
        if count >= min_freq_thresh:
            out_lines.append(f"{word}\t{i}\t{count}\n")
            words.append(word)
        else:
            unknown_count += count

    out_lines.insert(0, f"{'<unk>'}\t{0}\t{unknown_count}\n")

    with open('vocab.txt', 'w') as file:
        file.writelines(out_lines)

    tagset = list(set(tags))

    print(f"Number of words in vocabulary: {len(words) - 1}")
    print(f"Number of occurrences of <unk>: {unknown_count}")
    return lines, tagset, words


def generate_emission_matrix(lines, tagset, words):
    """
    Function to calculate the emission matrix from the corpus
    :param lines: list
    :param tagset: list
    :param words: list
    :return: np.ndarray
    """
    num_tags = len(tagset)
    num_words = len(words)

    # Allocate memory for the emission matrix
    emission_matrix = np.zeros((num_tags, num_words))

    for line in tqdm(lines):
        if len(line) == 3:
            _, word, tag = line
            word_idx = words.index(word) if word in words else 0
            emission_matrix[tagset.index(tag)][word_idx] += 1
            pass
        else:
            pass

    # Normalize emission matrix
    emission_matrix = np.divide(emission_matrix,
                                np.sum(emission_matrix, axis=1)[:, np.newaxis])

    # Pretty print emission matrix
    with pd.option_context('display.max_columns', None):
        em_df = pd.DataFrame(emission_matrix, index=tagset, columns=words)
        if DEBUG:
            print(em_df)
        em_df.to_csv('em.csv')

    return emission_matrix


def generate_transition_matrix(lines, tagset):
    """
    Function to generate the transition matrix for the given sentences and
    tagset
    :param lines: list
    :param tagset: list
    :return: np.ndarray
    """
    num_tags = len(tagset)

    # Allocate memory for the transition matrix
    transition_matrix = np.zeros((num_tags + 1, num_tags))
    start_word = [None, None, START_TAG]

    for word1, word2 in tqdm(zip([start_word] + lines, lines)):
        if len(word1) == 1:
            word1 = start_word
        if len(word2) == 3:
            _, _, tag_1 = word1
            _, _, tag_2 = word2
            row_idx = 0 if tag_1 == START_TAG else tagset.index(tag_1) + 1
            transition_matrix[row_idx][tagset.index(tag_2)] += 1

    # Normalize transition matrix
    transition_matrix = np.divide(transition_matrix,
                                  np.sum(transition_matrix, axis=1)[:,
                                  np.newaxis])

    # Pretty print transition matrix
    with pd.option_context('display.max_columns', None):
        tm_df = pd.DataFrame(transition_matrix,
                             index=[START_TAG] + tagset,
                             columns=tagset)
        if DEBUG:
            print(tm_df)
        tm_df.to_csv('tm.csv')

    return transition_matrix


class HiddenMarkovModel:
    def __init__(self):
        self.emission_matrix = None
        self.transition_matrix = None
        self.lines = None
        self.tagset = None
        self.words = None
        pass

    def fit(self, lines, tagset, words):
        # Assign member variables
        self.lines = lines
        self.tagset = tagset
        self.words = words

        self.emission_matrix = generate_emission_matrix(lines, tagset, words)
        self.transition_matrix = generate_transition_matrix(lines, tagset)
        return

    def predict(self, sentence, method='viterbi'):
        if method == 'viterbi':
            return self.viterbi_decoding(sentence)
        elif method == 'greedy':
            return self.greedy_decoding(sentence)
        else:
            raise NotImplementedError('Only greedy/viterbi decoding supported')

    def greedy_decoding(self, words):
        """
        Greedy decoding algorithm for decoding tags for a given sentence.
        :param write_output: bool
        :param words: str
        :return: list
        """
        output = []

        prev_tag_idx = None

        for idx, word in enumerate(words):
            word_idx = self.words.index(word) if word in self.words else 0
            # Handle first word in the sentence separately
            if idx == 0:
                # Calculate probabilities for each tag
                pred_prob = self.transition_matrix[0, :] * \
                            self.emission_matrix[:, word_idx]

            else:
                # Calculate probabilities for each tag
                pred_prob = self.transition_matrix[prev_tag_idx + 1, :] * \
                            self.emission_matrix[:, word_idx]

            # Get the tag with the maximum probability
            pred_idx = np.argmax(pred_prob)
            output.append(self.tagset[pred_idx])

            # Update previous tag
            prev_tag_idx = pred_idx

        return output

    def viterbi_decoding(self, words):
        # Allocate memory for the viterbi matrix
        # words = sentence.split(' ')
        # viterbi_mat = np.zeros((len(words), len(self.tagset)))
        viterbi_mat = np.zeros((len(self.tagset), len(words)))
        backpointer = np.zeros((len(self.tagset), len(words)), dtype=int)

        # Handle the first tag separately
        word_idx = self.words.index(words[0]) if words[0] in self.words else 0
        viterbi_mat[:, 0] = self.transition_matrix[0, :] * \
                            self.emission_matrix[:, word_idx]
        backpointer[:, 0] = 0

        # Perform viterbi decoding on the rest of the sentence
        for idx, word in enumerate(words[1:], start=1):
            word_idx = self.words.index(word) if word in self.words else 0
            for tag_idx, tag in enumerate(self.tagset):
                state_proba = viterbi_mat[:, idx - 1] * \
                              (self.transition_matrix[1:, tag_idx]) * \
                              (self.emission_matrix[tag_idx, word_idx])
                viterbi_mat[tag_idx, idx] = np.nanmax(state_proba)
                try:
                    backpointer[tag_idx, idx] = np.nanargmax(state_proba)
                except ValueError:
                    backpointer[tag_idx, idx] = -100

        best_path_prob = np.nanmax(viterbi_mat[:, len(words) - 1])
        best_path_pointer = np.nanargmax(viterbi_mat[:, len(words) - 1])

        # Trace the best path
        best_path = [self.tagset[best_path_pointer]]

        for i in range(backpointer.shape[1] - 1, 0, -1):
            best_path.append(self.tagset[backpointer[best_path_pointer, i]])
            best_path_pointer = backpointer[best_path_pointer, i]

        # Since we trace the path from behind, reverse the best path
        best_path.reverse()

        return best_path

    def save_model(self):
        train_params = {
            'emission': {},
            'transition': {}
        }

        # Create the emission dictionary
        for word_idx, word in enumerate(self.words):
            for tag_idx, tag in enumerate(self.tagset):
                if self.emission_matrix[tag_idx, word_idx] > 0:
                    train_params['emission'].__setitem__(
                        f'({tag}, {word})',
                        self.emission_matrix[tag_idx, word_idx]
                    )

        print(f"Number of emission parameters: {len(train_params['emission'])}")

        # Create the transition dictionary
        for tag_idx1, tag1 in enumerate([START_TAG] + self.tagset):
            for tag_idx2, tag2 in enumerate(self.tagset):
                if self.transition_matrix[tag_idx1, tag_idx2] > 0:
                    train_params['transition'].__setitem__(
                        f'({tag1}, {tag2})',
                        self.transition_matrix[tag_idx1, tag_idx2]
                    )
        print(f"Number of transition parameters: "
              f"{len(train_params['transition'])}")

        with open('hmm.json', 'w') as file:
            json.dump(train_params,
                      file,
                      indent=4)


def get_predictions(filepath, model, no_targets=False, **kwargs):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    lines = [line.rstrip('\n') for line in lines]
    lines = [line.split('\t') for line in lines]

    dataset = []
    datum = []

    if not no_targets:
        target = []  # To store the tags for a single sentence
        targets = []  # To store the tags for the complete dataset

    for line in lines:
        if len(line) == 1:
            dataset.append(datum)
            datum = []

            if not no_targets:
                targets.append(target)
                target = []
        else:
            datum.append(line[1])

            if not no_targets:
                target.append(line[2])

    dataset.append(datum)
    if not no_targets:
        targets.append(target)

    outputs = []

    for datum in tqdm(dataset):
        outputs.append(model.predict(datum, kwargs.get('method')))

    if not no_targets:
        targets = list(chain(*targets))
        outputs = list(chain(*outputs))

        assert len(targets) == len(outputs)

        correct = len([i for i in range(len(outputs)) if
                       outputs[i] == targets[i]])

        print(f"Accuracy on the dev set using {kwargs['method']}: "
              f"{correct / len(outputs)}")
    else:
        # Write outputs to out file
        with open(f"{kwargs.get('method')}.out", 'w') as file:
            for data_idx, datum in enumerate(dataset):
                for idx, word in enumerate(datum):
                    file.write(f"{idx + 1}\t{word}\t{outputs[data_idx][idx]}\n")
                file.write('\n')

    return outputs


if __name__ == '__main__':
    lines, tagset, words = read_data(TRAIN_DATA)
    hmm = HiddenMarkovModel()
    hmm.fit(lines, tagset, words)
    hmm.save_model()

    get_predictions(VAL_DATA, model=hmm, method='viterbi')
    get_predictions(VAL_DATA, model=hmm, method='greedy')
    get_predictions(TEST_DATA, model=hmm,
                    no_targets=True, method='viterbi')
    get_predictions(TEST_DATA, model=hmm,
                    no_targets=True, method='greedy')
