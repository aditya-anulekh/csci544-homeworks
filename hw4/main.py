import os
import sys
import numpy as np
import torch
import torch.nn as nn
import config as cfg
from models import BLSTM
from utils import (
    get_dataloaders,
    train_model,
    read_data,
    load_glove_vec,
    generate_outputs
)

PROJECT_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
TRAIN_DATA = os.path.join(DATA_ROOT, 'train')
DEV_DATA = os.path.join(DATA_ROOT, 'dev')
TEST_DATA = os.path.join(DATA_ROOT, 'test')
GLOVE_PATH = os.path.join(PROJECT_ROOT, 'glove.6B.100d.txt')
SAVED_MODELS_PATH = os.path.join(PROJECT_ROOT, 'saved_models')


def part_1():
    _, vocab, tagset, tag_weight = read_data(TRAIN_DATA)
    tag_weight = torch.Tensor(tag_weight).to(cfg.DEVICE)

    train_dataloader, val_dataloader = get_dataloaders(TRAIN_DATA,
                                                       vocab=vocab,
                                                       tagset=tagset)

    model = BLSTM(len(vocab), len(tagset), embedding_dim=100)

    criterion = nn.CrossEntropyLoss(weight=tag_weight, ignore_index=0)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE,
                            momentum=0.1)

    model, metrics = train_model(
        model=model,
        optimizer=optim,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=10,
    )
    torch.save(model, os.path.join(SAVED_MODELS_PATH, 'blstm1.pt'))

    model.eval()
    generate_outputs(model, DEV_DATA, 'output.txt', connl_eval=True,
                     vocab=vocab, tagset=tagset)

    # model = torch.load(os.path.join(SAVED_MODELS_PATH, 'blstm1.pt'))
    # model.eval()
    #
    # generate_outputs(model, DEV_DATA, 'dev1.out', connl_eval=True,
    #                  vocab=vocab, tagset=tagset)
    return


def part_2():
    _, _, tagset, tag_weight = read_data(TRAIN_DATA)
    glove_vec = load_glove_vec(GLOVE_PATH)
    word_embeddings = torch.Tensor(np.vstack(list(glove_vec.values())))

    tag_weight = torch.Tensor(tag_weight).to(cfg.DEVICE)

    train_dataloader, val_dataloader = get_dataloaders(
        TRAIN_DATA, vocab=list(glove_vec.keys()), tagset=tagset
    )

    model = BLSTM(
        vocab_size=len(list(glove_vec.keys())),
        tagset_size=len(tagset),
        embedding_dim=100,
        word_embeddings=word_embeddings
    )

    criterion = nn.CrossEntropyLoss(weight=tag_weight, ignore_index=0)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE,
                            momentum=0.1)

    model, metrics = train_model(
        model=model,
        optimizer=optim,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=30,
    )

    torch.save(model, os.path.join(SAVED_MODELS_PATH, 'blstm2.pt'))

    generate_outputs(model, DEV_DATA, 'output.txt',
                     vocab=list(glove_vec.keys()), tagset=tagset)
    return


def inference():
    # TODO: Add inference for test set
    # Inference for part 1
    _, vocab, tagset, _ = read_data(TRAIN_DATA)

    model = torch.load(os.path.join('saved_models', 'blstm1.pt'))
    model.eval()

    generate_outputs(model, DEV_DATA, 'dev1.out', connl_eval=False,
                     vocab=vocab, tagset=tagset)

    generate_outputs(model, TEST_DATA, 'test1.out', connl_eval=False,
                     vocab=vocab, tagset=tagset, no_targets=True)

    # Inference for part 2
    _, _, tagset, _ = read_data(TRAIN_DATA)
    glove_vec = load_glove_vec(GLOVE_PATH)

    model = torch.load(os.path.join('saved_models', 'blstm2.pt'))
    model.eval()

    generate_outputs(model, DEV_DATA, 'dev2.out', connl_eval=False,
                     vocab=list(glove_vec.keys()), tagset=tagset)
    pass


if __name__ == '__main__':
    args = sys.argv
    assert len(args) == 2, "Must provide at least one action from " \
                           "part_1, part_2, inference"

    if args[1] == 'part_1':
        part_1()
    elif args[1] == 'part_2':
        part_2()
    elif args[1] == 'inference':
        inference()
