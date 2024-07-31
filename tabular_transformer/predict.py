"""
predict from the trained model with PyTorch
"""
from contextlib import nullcontext
import torch
from .preprocessor import preprocess
from .tabular_transformer import ModelArgs, TabularTransformer
from .tokenizer import Tokenizer
from .dataloader import load_data
import random
from .util import LossType
import numpy as np
from .metrics import calAUC


# -----------------------------------------------------------------------------
checkpoint = 'out/ckpt.pt'

predict_dataset = 'income/income_evaluation_validate.csv'
has_truth = True

batch_size = 128
seed = 1337

# examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
compile = False  # use PyTorch 2.0 to compile the model to be faster
# overrides from command line or config file
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

rng = random.Random(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# for later use in torch.autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
dataset_attr = checkpoint_dict['dataset_attr']
train_config = checkpoint_dict['config']
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = TabularTransformer(gptconf)

state_dict = checkpoint_dict['model']

unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)

# load the tokenizer
enc = Tokenizer(dataset_attr['feature_vocab'], dataset_attr['feature_type'])

target_map = dataset_attr['target_map']
assert target_map is not None
loss_type = train_config['loss_type']
assert LossType(
    loss_type) is LossType.BINCE, "only support binary cross entropy loss"

predict_map = {v: k for k, v in target_map.items()}

predict_dataframe = load_data(predict_dataset)

if has_truth:
    dataset_x = predict_dataframe.iloc[:, :-1]
    truth_y = predict_dataframe.iloc[:, -1]

else:
    dataset_x = predict_dataframe
    truth_y = None
assert dataset_x.shape[1] == dataset_attr['num_cols']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


accuracy_accum = {"n_samples": 0, "right": 0}


def get_results(logits_arr):
    prob_y = sigmoid(logits_arr)
    result_val = np.where(prob_y > 0.5, 1, 0)
    result_cls = np.vectorize(predict_map.get)(result_val)
    return result_cls


def accum_accuracy(logits, truth=None):
    if truth is None:
        return
    assert len(logits) == len(truth)
    result_cls = get_results(logits)
    equal_elements = np.sum(np.equal(result_cls, truth))
    accuracy_accum['n_samples'] += len(logits)
    accuracy_accum['right'] += equal_elements


def binary_cross_entropy_loss(logits, targets=None):
    # Apply sigmoid to logits
    probs = sigmoid(logits)
    targets = np.vectorize(target_map.get)(targets)
    # Compute binary cross-entropy loss
    loss = - (targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
    # Return the mean loss
    return np.mean(loss)


logits_array = np.zeros(len(dataset_x), dtype=float)

# run generation
with torch.no_grad():
    with ctx:
        num_batches = (len(dataset_x) + batch_size - 1) // batch_size
        for ix in range(num_batches):
            # encode the beginning of the prompt
            start = ix * batch_size
            end = start + batch_size

            x = dataset_x[start: end]

            truth = truth_y[start: end] if truth_y is not None else None

            # preprocess the data
            xp = preprocess(rng,
                            x,
                            dataset_attr['feature_type'],
                            dataset_attr['feature_stats'],
                            train_config['apply_power_transform'],
                            train_config['remove_outlier'],
                            )
            tok_x = enc.encode(xp)
            feature_tokens = tok_x[0].to(device, non_blocking=True)
            feature_weight = tok_x[1].to(device, non_blocking=True)
            logits = model.predict((feature_tokens, feature_weight))
            logits_y = logits.squeeze(-1).to('cpu').numpy()
            # save in the array
            logits_array[start: end] = logits_y

            accum_accuracy(logits_y, truth.to_numpy()
                           if truth is not None else None)

predict_result_array = get_results(logits_array)

if truth_y is not None:
    bce_loss = binary_cross_entropy_loss(logits_array, truth_y.to_numpy())
    print(f"binary cross entropy loss: {bce_loss:.4f}")
    auc_score = calAUC(truth_y.to_numpy(), logits_array)
    print(f"auc score: {auc_score}")
    print(f"samples: {accuracy_accum['n_samples']}, "
          f"accuracy: {accuracy_accum['right'] / accuracy_accum['n_samples']:.2f}")
