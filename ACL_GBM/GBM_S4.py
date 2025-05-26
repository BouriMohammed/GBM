import numpy as np
import argparse
import os
import utils_s4 as utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import logging
import colorlog
import warnings
import gzip
warnings.filterwarnings("ignore")

# Fix the seed for reproducibility


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        cudnn.deterministic = True  # Ensures deterministic behavior
        cudnn.benchmark = False     # Disables benchmark for deterministic runs


# Configure colorlog for colored logging
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'CRITICAL': 'cyan',  # Cyan for loading/importing actions
        'INFO': 'green',  # Green for done/completed tasks
        'DEBUG': 'red',
    }
)
handler.setFormatter(formatter)
logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--hidden_size', '-d', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--num_epochs', '-T', type=int, default=1)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--checkpoint',  action='store_true', default=False)
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Num blocks of S4')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='Num of classes')
    parser.add_argument('--portion', '-k', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--portion_egv', '-kegv', type=float, default=0.0)
    parser.add_argument('--portiona', '-ka', type=float, default=0.0)
    parser.add_argument('--portionc', '-kc', type=float, default=0.0)
    parser.add_argument('--portionb', '-kb', type=float, default=0.0)
    args = parser.parse_args()
    return args


args = parse_args()
batch_size = args.batch_size
checkpoint = args.checkpoint
num_epochs = args.num_epochs
d_output = args.num_classes
d_model = args.hidden_size
n_layers = args.num_layers
seed = args.seed
max_len = args.max_len
k_egv = args.portion_egv
kc = args.portionc
kb = args.portionb
ka = args.portiona
k = args.portion
lr = args.learning_rate
patience = args.patience


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_loss = float('inf')  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
patience = patience
counter = 0
set_seed(seed)

# loading Data vectors
logger.critical('Loading data...')


xtrain = np.load('./data/imdb/xtrain.npy')
xtest = np.load('./data/imdb/xtest.npy')

y_train = np.load('./data/imdb/y_train.npy')
y_test = np.load('./data/imdb/y_test.npy')

logger.info('Loading Data DONE !!')

# loading Embedding vector
logger.critical('Loading Embedding Matrix...')
embedding_matrix = np.load('./data/embedding_matrix.npy')

logger.info('Loading Embedding matrix DONE!!')

train_dataset = utils.IMDBDataset(reviews=xtrain, targets=y_train)
valid_dataset = utils.IMDBDataset(reviews=xtest, targets=y_test)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False)
valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False)


# Model
logger.critical('==> Building model..')
model = utils.S4Model(
    embedding_matrix=embedding_matrix,
    d_input=embedding_matrix.shape[1],
    d_output=d_output,
    d_model=d_model,
    n_layers=n_layers
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim")
           for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(
            p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler
    # return optimizer


optimizer, scheduler = setup_optimizer(
    model, lr=lr, weight_decay=0.01, epochs=num_epochs)


eng = utils.Engine_gbm_discret_derivative(
    model, optimizer, scheduler, ka, kb, device)


logger.critical('Training phase\n')
for epoch in range(num_epochs):
    print(f'----------- Epoch {epoch+1} -----------\n')

    eng.train(epoch, train_data_loader, num_epochs)

    # Run evaluation on validation set (if any)
    avg_loss = eng.eval(epoch, valid_data_loader, num_epochs, checkpoint=True)

    # Update scheduler for the next epoch
    scheduler.step()
    # Optionally print the learning rate at the end of each epoch
    print(f"Epoch {epoch+1} learning rate: {scheduler.get_last_lr()[0]}")

    # Save checkpoint if accuracy improves
    if checkpoint:
        if avg_loss <= best_loss:

            best_model_state = model.state_dict()

            os.makedirs(f'./model/imdb/S4/{ka}_{kb}', exist_ok=True)
            with gzip.open(f'./model/imdb/S4/{ka}_{kb}/model.pth.gz', 'wb') as f:
                torch.save(best_model_state, f)

            best_loss = avg_loss
            logger.critical('Saving BEST model')
            counter = 0
        else:
            counter += 1
            logger.debug(f'EarlyStopping step: {counter}')
            # print('EarlyStopping step: ', counter)
            if counter >= patience:
                print(
                    f'Early stopping at epoch {epoch + 1} as no improvement was seen in validation loss. MUST STOP HERE !!!')
                break
