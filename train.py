import torch
from torch.utils.data import DataLoader

import time
import pickle
import os
import logging
import json
import argparse

from utils.process_data import process_data
from model_fn import CBOWHierSoftmax, CBOWNegativeSampling
from input_fn import CBOWDataSet
from utils.utils import set_logger, str2bool, Params, make_directory


def train_fn(params, loader, device):
    if params.model == 'neg_sampling':
        model = CBOWNegativeSampling(
            emb_count=params.emb_count,
            emb_dim=params.emb_dim,
            neg_sampling_factor=params.neg_samples,
            device=device
        )
    elif params.model == 'hier_softmax':
        model = CBOWHierSoftmax(
            emb_count=params.emb_count,
            emb_dim=params.emb_dim,
            cbow_op=params.cbow_op,
            emb_max_norm=params.emb_max_norm,
            emb_norm_type=params.emb_norm_type,
        )
    else:
        raise NotImplementedError(f'Wrong param model: {params.model}')

    if 'cuda' in params.device:
        model.cuda(device=params.device)

    if params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        raise NotImplementedError(f'{params.optimizer} is not supported!')

    epochs = params.epochs

    use_scheduler = False
    if getattr(params, 'lr_drop_factor', None) is not None:
        use_scheduler = True
        schedule_fn = lambda epoch: params.lr_drop_factor ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=[schedule_fn],
                                                      last_epoch=-1)
    st = time.time()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        iterator = loader
        i = 0
        # Print Learning Rate
        if use_scheduler:
            _lr = scheduler.get_lr()[0]
        else:
            _lr = params.lr
        logging.info(f'Epoch: {epoch}; LR: {_lr}')
        for input in iterator:
            i += 1
            input = [i.to(device) for i in input]
            model.zero_grad()
            loss = model(*input)

            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            if loss_val < 0:
                raise Exception('loss_val < 0')
            total_loss += loss_val
            if i % params.log_freq == 0 and i > 0:
                logging.info(
                    f'epoch: {epoch}; step: {i}; '
                    f'mean loss: {total_loss/i:.5f}; '
                    f'elapsed time: {time.time() - st:.2f}s')
            fstep = i

        folder = \
            f'{FLAGS.folder}/ckpt'
        if not os.path.exists(folder):
            os.mkdir(folder)

        savepath =\
            f'{folder}/e{epoch}-{params.batch_size}-lr{_lr:.5f}-' \
            f'loss{total_loss/fstep:.5f}-w2vec-bible.ckpt.tar'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss/fstep,
            'lr':   _lr
        }, savepath)

        if use_scheduler:
            scheduler.step()

        logging.info(
            f'epoch: {epoch}; mean total_loss: {total_loss/fstep:.5f}; '
            f'elapsed time: {time.time() - st:.2f}s')
        losses.append(total_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=str, default='/tmp/',
                        help='async/not-async requests')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='fullpath to conf.json')
    parser.add_argument('--datapath', '-d', type=str,
                        default='./data/bible.txt')
    FLAGS = parser.parse_args()

    if not os.path.exists(FLAGS.folder):
        make_directory(FLAGS.folder)

    set_logger(os.path.join(FLAGS.folder, 'train.log'))
    if FLAGS.config is None:
        try:
            FLAGS.config = os.path.join(FLAGS.folder, 'config.json')
        except FileNotFoundError:
            raise FileNotFoundError('config.json is not found!')
    params = Params(jsonpath=FLAGS.config)

    logging.info('Start word2vec training pipeline! Params:')
    logging.info(
        json.dumps(params.__dict__, indent=True)
    )

    if params.model not in ['hier_softmax', 'neg_sampling']:
        raise NotImplementedError(f"{params.model} model is not supported!")

    # load data:
    logging.info('Loading data:')

    processed_datapath = os.path.join(FLAGS.folder,
                                      f'{params.model}_processed_data.pkl')

    processing_params = dict(
        threshold_count=params.threshold_count,
        pipeline=params.model,
        downsampling_params=getattr(params, 'downsampling_params', None)
    )
    try:
        loaded_params, out = \
            pickle.load(open(processed_datapath, 'rb'))
        if loaded_params != processing_params:
            raise FileNotFoundError
    except FileNotFoundError:
        out = process_data(FLAGS.datapath, **processing_params)
        pickle.dump([processing_params, out],
                    open(processed_datapath, 'wb'))
    if params.model == 'neg_sampling':
        corpus, words_hash_inversed, vocab_size = out
        params.emb_count = vocab_size
    elif params.model == 'hier_softmax':
        corpus, node_inxs, turns_inxs, leaves_hash_inversed, \
            vocab_size, nodes_count = out
        params.emb_count = nodes_count

    device = torch.device(params.device)

    if params.model == 'neg_sampling':
        cbow_dataset = CBOWDataSet(corpus,
                                   pipeline=params.model,
                                   neg_samples=params.neg_samples,
                                   window_size=params.window_size,
                                   device=None)
    elif params.model == 'hier_softmax':
        cbow_dataset = CBOWDataSet(corpus,
                                   pipeline=params.model,
                                   nodes_index=node_inxs,
                                   turns_index=turns_inxs,
                                   vocab_size=nodes_count,
                                   window_size=params.window_size,
                                   skip_target=True,
                                   device=None)

    data_len = cbow_dataset.__len__()
    n_steps  = (data_len - 1) // params.batch_size
    loader = DataLoader(cbow_dataset, batch_size=params.batch_size,
                        shuffle=False, num_workers=params.num_workers)
    train_fn(params, loader, device)




