import os
import math
import numpy as np
import torch.nn
from tqdm import tqdm
from torch.utils.data import (
    Dataset, DataLoader, random_split, SequentialSampler
)
from sklearn.utils.class_weight import compute_class_weight
import config as cfg


def read_data(filepath, train_data=True):
    min_freq_thresh = 1
    with open(filepath, 'r') as file:
        lines = file.readlines()

    lines = [line.rstrip('\n') for line in lines]
    lines = [line.split(' ') for line in lines]

    # Creating a vocab file
    counts = {}
    tags = []
    for line in tqdm(lines):
        if len(line) == 3:
            _, word, tag = line
            word = word.lower()
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

    # out_lines = []
    unknown_count = 0

    for i, (word, count) in enumerate(counts.items(), start=1):
        if count >= min_freq_thresh:
            # out_lines.append(f"{word}\t{i}\t{count}\n")
            words.append(word)
        else:
            unknown_count += count

    # out_lines.insert(0, f"{'<unk>'}\t{0}\t{unknown_count}\n")

    # with open('vocab.txt', 'w') as file:
    #     file.writelines(out_lines)

    tagset = list(set(tags))
    tag_weights = compute_class_weight(class_weight='balanced',
                                       classes=np.array(tagset), y=tags)
    tag_weights = np.insert(tag_weights, 0, 0)
    tagset.insert(0, '<PAD>')

    print(f"Number of words in vocabulary: {len(words) - 1}")
    print(f"Number of occurrences of <unk>: {unknown_count}")
    return lines, words, tagset, tag_weights


def load_glove_vec(filepath):
    glove_vec = {'<unk>': np.zeros(100)}
    with open(filepath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.split(' ')
        glove_vec.__setitem__(
            line[0],
            np.asarray(line[1:], dtype='float32')
        )

    return glove_vec


class NERDataset(Dataset):
    def __init__(self, filepath, vocab, tagset, no_targets=False):
        self.filepath = filepath            # Path of the datafile
        self.no_targets = no_targets        # True for test data
        self.vocab = vocab                  # Unique words
        self.tagset = tagset                # Unique tags

        self.sentences, self.targets = self.get_sentences()
        self.max_len = len(max(self.sentences, key=len))
        self.vocab_map = {word: i for (i, word) in enumerate(self.vocab)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Create a vector to store the case
        case_bool = torch.Tensor([
            0 if i.lower() == i else 1 for i in sentence
        ])

        # Convert words to indices
        sentence = torch.LongTensor([self.vocab_map.get(i.lower(), 0)
                                     for i in sentence])

        # Calculate the padding length and pad the sentence
        n_words = len(sentence)
        # pad_length = self.max_len - n_words
        # padding = torch.nn.ConstantPad1d((0, pad_length), 0)
        # sentence = padding(sentence)

        if not self.no_targets:
            targets = self.targets[idx]
            targets = torch.Tensor([self.tagset.index(i) for i in targets])
            # targets = padding(targets)
            return (sentence, case_bool, n_words), \
                targets.type(torch.LongTensor)
        else:
            return (sentence, case_bool, n_words), \
                None

    def get_sentences(self):
        with open(self.filepath, 'r') as file:
            lines = file.readlines()

        lines = [line.rstrip('\n') for line in lines]
        lines = [line.split(' ') for line in lines]

        dataset = []
        datum = []

        target = []  # To store the tags for a single sentence
        targets = []  # To store the tags for the complete dataset

        for line in lines:
            if len(line) == 1:
                dataset.append(datum)
                datum = []

                if not self.no_targets:
                    targets.append(target)
                    target = []
            else:
                datum.append(line[1])

                if not self.no_targets:
                    target.append(line[2])

        dataset.append(datum)
        if not self.no_targets:
            targets.append(target)

        return dataset, targets


def collate_fn(data):
    """
    Function to pad sentences to the max length in that batch
    :param data:
    :return:
    """
    max_len = max([l for (_, _, l), _ in data])
    batch_size = len(data)
    sentences_batched = torch.zeros((batch_size, max_len), dtype=torch.long)
    case_batched = torch.zeros((batch_size, max_len), dtype=torch.bool)
    lengths_batched = []
    targets_batched = torch.zeros((batch_size, max_len), dtype=torch.long)

    # TODO: Fix for cases that don't have a target
    for i, ((sentence, case_bool, length), target) in enumerate(data):
        # Calculate the padding length and pad the sentence
        pad_length = max_len - length
        padding = torch.nn.ConstantPad1d((0, pad_length), 0)
        sentence = padding(sentence)
        target = padding(target)
        case_bool = padding(case_bool)

        sentences_batched[i, :] = sentence
        case_batched[i, :] = case_bool
        lengths_batched.append(length)
        targets_batched[i, :] = target

    sentences_batched = torch.Tensor(sentences_batched)
    case_batched = torch.Tensor(case_batched)
    lengths_batched = torch.Tensor(lengths_batched)
    targets_batched = torch.Tensor(targets_batched)

    return (sentences_batched, case_batched, lengths_batched), targets_batched


def get_dataloaders(train_data, split=True, **kwargs):

    train_dataset = NERDataset(train_data, kwargs['vocab'], kwargs['tagset'])
    # val_dataset = NERDataset(val_data, kwargs['vocab'], kwargs['tagset'])
    # test_dataset = NERDataset(test_data, kwargs['vocab'], kwargs['tagset'],
    #                           no_targets=True)

    if split:
        train_len = math.floor(0.8 * len(train_dataset))
        val_len = len(train_dataset) - train_len
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_len, val_len],
            torch.Generator().manual_seed(cfg.RANDOM_SEED))
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
    )
    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
    #                              shuffle=False)
    if split:
        return train_dataloader, val_dataloader
    else:
        return train_dataloader, None


def train_model(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        num_epochs=30,
        lr_scheduler=None
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        metrics = {
            'train_acc': 0,
            'train_loss': 0.0,
            'val_acc': 0,
            'val_loss': 0.0
        }

        for i, ((X, case, lengths), y) in enumerate(tqdm(train_dataloader)):
            model.train()
            # Zero optim gradients
            optimizer.zero_grad()

            # Move to GPU
            X = X.to(device)                    # (batch_size, seq_len)
            y = y.to(device)                    # (batch_size, num_cls)
            case = case.to(device)

            # Forward pass
            outputs = model(X, case, lengths)   # (batch_size, seq_len, num_cls)
            outputs = outputs.permute(0, 2, 1)  # (batch_size, num_cls, seq_len)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print(f"{torch.argmax(outputs, axis=1) = }")
            # print(f"{y = }")

            # Calculate the accuracy
            metrics['train_acc'] += (
                        torch.argmax(outputs, axis=1) == y).float().sum()

            # Calculate the loss
            metrics['train_loss'] += loss

        metrics['train_acc'] /= (
                    len(train_dataloader) * train_dataloader.batch_size)
        metrics['train_loss'] /= (
                    len(train_dataloader) * train_dataloader.batch_size)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # for i, (X, y) in enumerate(tqdm(val_dataloader)):
        #     model.eval()
        #
        #     # Move to GPU
        #     X = X.to(device)
        #     y = y.to(device)
        #
        #     # Forward pass
        #     outputs = model(X)
        #     loss = criterion(outputs, y)
        #
        #     # Calculate the accuracy
        #     metrics['val_acc'] += (
        #                 torch.argmax(outputs, axis=1) == y).float().sum()
        #
        #     # Calculate the loss
        #     metrics['val_loss'] += loss
        #
        # metrics['val_acc'] /= (len(val_dataloader) * val_dataloader.batch_size)
        # metrics['val_loss'] /= (len(val_dataloader) * val_dataloader.batch_size)

        print(f"Epoch: {epoch + 1}/{num_epochs}")
        print("Mode\tLoss\tAcc")
        print(f"Train\t{metrics['train_loss']:.2f}\t{metrics['train_acc']:.2f}")
        print(f"Valid\t{metrics['val_loss']:.2f}\t{metrics['val_acc']:.2f}")

    return model, metrics


def generate_outputs(model, test_file, out_file,
                     no_targets=False, connl_eval=False, **kwargs):
    # Remove the out file if it already exists
    try:
        os.remove(out_file)
    except FileNotFoundError:
        pass

    vocab = kwargs['vocab']
    tagset = kwargs['tagset']
    test_dataset = NERDataset(test_file, vocab,
                              tagset, no_targets)
    sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                                 shuffle=False, collate_fn=collate_fn,
                                 sampler=sampler)

    model = model.to(cfg.DEVICE)

    for i, ((X, case_bool, lengths), y) in enumerate(tqdm(test_dataloader)):
        model.eval()

        # Move to GPU
        X = X.to(cfg.DEVICE)  # (batch_size, seq_len)
        y = y.to(cfg.DEVICE)  # (batch_size, num_cls)
        case_bool = case_bool.to(cfg.DEVICE)

        output = model(X, case_bool, lengths)
        output = torch.argmax(output, axis=2)

        with open(out_file, 'a') as file:
            for j in range(len(output)):
                for k in range(int(lengths[j])):
                    sentence_idx = i * cfg.BATCH_SIZE + j
                    if connl_eval:
                        file.write(
                            f'{k + 1} {test_dataset.sentences[sentence_idx][k]}'
                            f' {tagset[y[j][k]]} {tagset[output[j][k]]}\n')
                    else:
                        file.write(
                            f'{k + 1} {test_dataset.sentences[sentence_idx][k]}'
                            f' {tagset[output[j][k]]}\n')
                file.write('\n')

    pass


if __name__ == '__main__':
    # train_file = 'data/train'
    # l, w, t, _ = read_data(train_file)
    # dataset = NERDataset(train_file, w, t)
    #
    # tl, vl = get_dataloaders(train_file, vocab=w, tagset=t)
    # print(next(iter(tl)).shape)
    # print(next(iter(vl)).shape)
    gv = load_glove_vec('glove.6B.100d.txt')
    print(torch.Tensor(np.vstack(list(gv.values()))).shape)
    print(gv['<unk>'])
