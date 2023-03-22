import os
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
        num_epochs=5,
    )
    torch.save(model.state_dict(), os.path.join(SAVED_MODELS_PATH, 'blstm1.pt'))

    generate_outputs(model, DEV_DATA, 'output.txt', vocab=vocab, tagset=tagset)
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

    torch.save(model.state_dict(), os.path.join(SAVED_MODELS_PATH, 'blstm2.pt'))

    generate_outputs(model, DEV_DATA, 'output.txt',
                     vocab=list(glove_vec.keys()), tagset=tagset)
    return


if __name__ == '__main__':
    part_1()
