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
    _, vocab, tag_to_idx, idx_to_tag, tag_weight = read_data(TRAIN_DATA)
    tag_weight = torch.Tensor(tag_weight).to(cfg.DEVICE)
    num_tags = len(tag_to_idx)

    train_dataloader, val_dataloader = get_dataloaders(
        TRAIN_DATA, vocab=vocab, tagset=tag_to_idx, batch_size=cfg.BATCH_SIZE_1)

    model = BLSTM(len(vocab), num_tags, embedding_dim=100)

    criterion = nn.CrossEntropyLoss(weight=tag_weight, ignore_index=-1)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE_1,
                            momentum=0.1)

    model, metrics = train_model(
        model=model,
        optimizer=optim,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=20,
    )
    torch.save(model, os.path.join(SAVED_MODELS_PATH, 'blstm1.pt'))

    return


def part_2():
    _, vocab, tag_to_idx, idx_to_tag, tag_weight = read_data(TRAIN_DATA)
    glove_vec = load_glove_vec(GLOVE_PATH)
    word_embeddings = torch.Tensor(np.vstack(list(glove_vec.values())))
    num_tags = len(tag_to_idx)

    tag_weight = torch.Tensor(tag_weight).to(cfg.DEVICE)

    train_dataloader, val_dataloader = get_dataloaders(
        TRAIN_DATA, vocab=list(glove_vec.keys()), tagset=tag_to_idx,
        batch_size=cfg.BATCH_SIZE_2
    )

    model = BLSTM(
        vocab_size=len(list(glove_vec.keys())),
        tagset_size=num_tags,
        embedding_dim=100,
        word_embeddings=word_embeddings
    )

    criterion = nn.CrossEntropyLoss(weight=tag_weight, ignore_index=-1)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE_2,
                            momentum=0.5)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optim, patience=2, factor=0.5)

    model, metrics = train_model(
        model=model,
        optimizer=optim,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=30,
        # lr_scheduler=lr_scheduler,
    )

    torch.save(model, os.path.join(SAVED_MODELS_PATH, 'blstm2.pt'))

    return


def inference():
    # Inference for part 1
    print("****** PART 1 ******")
    _, vocab, tag_to_idx, idx_to_tag, _ = read_data(TRAIN_DATA)

    model = torch.load(os.path.join(SAVED_MODELS_PATH, 'blstm1.pt'))
    print("INFO: Successfully loaded model")
    model.eval()

    print("INFO: Running inference on the dev set")
    generate_outputs(model, DEV_DATA, 'dev1.out', connl_eval=True,
                     vocab=vocab, idx_to_tag=idx_to_tag, tag_to_idx=tag_to_idx)

    print("INFO: Running inference on the test set")
    generate_outputs(model, TEST_DATA, 'test1.out', connl_eval=False,
                     vocab=vocab, idx_to_tag=idx_to_tag, tag_to_idx=tag_to_idx,
                     no_targets=True)

    # Inference for part 2
    print("****** PART 2 ******")
    _, vocab, tag_to_idx, idx_to_tag, _ = read_data(TRAIN_DATA)
    glove_vec = load_glove_vec(GLOVE_PATH)

    model = torch.load(os.path.join(SAVED_MODELS_PATH, 'blstm2.pt'))
    print("INFO: Successfully loaded model")
    model.eval()

    print("INFO: Running inference on the dev set")
    generate_outputs(model, DEV_DATA, 'dev2.out', connl_eval=True,
                     vocab=list(glove_vec.keys()), idx_to_tag=idx_to_tag,
                     tag_to_idx=tag_to_idx)

    print("INFO: Running inference on the test set")
    generate_outputs(model, TEST_DATA, 'test2.out', connl_eval=False,
                     vocab=vocab, idx_to_tag=idx_to_tag, tag_to_idx=tag_to_idx,
                     no_targets=True)
    return


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
