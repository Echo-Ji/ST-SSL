import os 
import argparse 
import h5py 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_st_dataset(dataset='BikeNYC'):
    """
    Load spatio-temporal dataset from the original dataset and output np.array \
    dataset with shape: (sequence_length, num_of_vertices, num_of_features).

    Parameters
    --------
    dataset: {'BikeNYC', 'TaxiBJ15'}, optional
        Dataset signature. The default is 'BikeNYC'.
    
    Returns
    --------
    data: np.array
        Loaded data.
    """
    # output shape: (sequence_length, num_of_vertices, num_of_features)
    if dataset == 'BikeNYC':
        # h5py to npz
        data_path =  'data/BikeNYC/NYC14_M16x8_T60_NewEnd.h5'
        f = h5py.File(data_path)
        data = np.array(f['data'])
        seq_length, feat, row, col = data.shape
        data = np.reshape(data, (seq_length, feat, row * col))
        data = np.transpose(data, (0, 2, 1))
    elif dataset == 'TaxiBJ15':
        # h5py to npz
        data_path =  'data/TaxiBJ/BJ15_M32x32_T30_InOut.h5'
        f = h5py.File(data_path)
        data = np.array(f['data'])
        seq_length, feat, row, col = data.shape
        data = np.reshape(data, (seq_length, feat, row * col))
        data = np.transpose(data, (0, 2, 1))
    else:
        # newly added dataset can be implemented here.
        raise ValueError
    
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        
    print('Load {} dataset with shape (seq_len, num_nodes, feat_dim): {}'.format(dataset, data.shape))
    return data

def gen_graph_seq2seq_data(data, x_offsets, y_offsets):
    """
    Generate samples from np.array data.

    Returns
    --------
    x: np.array 
        Input with shape (batch_size, input_length, num_nodes, input_dim).
    y: np.array 
        Output with shape (batch_size, output_length, num_nodes, output_dim).
    """
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    num_samples, num_nodes = data.shape[:2]

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    x, y = [], []
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def gen_train_val_test(args):
    data = load_st_dataset(args.dataset) # seq_length, num_nodes, feat_dim, (lvc)
    # input offsets
    day_size = args.day_size # n timeslice per day
    ts = 24 / day_size # time slice, e.g., 0.5 hour, 1 hour
    pw = int(args.period_width / ts) # period span contains how many time slices
    cn = int(args.closeness / ts) # closeness contains how many time slices
    period_centers = []
    for i in range(args.period):
        period_centers.append(1 - (i + 1) * day_size)
    x_offsets = np.sort(
        np.concatenate((
            *[np.arange(pc - pw, pc + pw + 1, 1) for pc in period_centers], 
            np.arange(1 - cn, 1, 1))
        )
    )
    # output offsets
    y_offsets = np.sort(np.arange(1, 1 + args.horizon, 1))
    print('Input length: {}, output length: {}\n'.format(len(x_offsets), len(y_offsets)))

    x, y = gen_graph_seq2seq_data(
        data, 
        x_offsets=x_offsets, 
        y_offsets=y_offsets, 
    )
    print('x shape: {}, y shape {}'.format(x.shape, y.shape))

    # Use 20% as testing data. For the rest, 7/8 is used for training while 1/8 for evaluation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    save_dir = os.path.join(args.output_dir, args.dataset)
    if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir, exist_ok=True)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, args.dataset, "%s.npz" % cat), 
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

def main(args):
    print('Generating training data')
    gen_train_val_test(args)

if __name__ == '__main__':
    '''
    # BikeNYC
    python scripts/generate_training_data.py --output_dir=data/ --dataset=BikeNYC --day_size=24 --closeness=4 --period_width=2

    # TaxiBJ15
    python scripts/generate_training_data.py --output_dir=data/ --dataset=TaxiBJ15 --day_size=48 --closeness=8 --period_width=4
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        '--dataset', type=str, default='BikeNYC', help='Dataset signature.'
    )
    parser.add_argument(
        '--day_size', type=int, default=24, help='Number of time slices per day.'
    )
    parser.add_argument(
        '--closeness', type=int, default=4, 
        help='Previous 4 hours.',
    )
    parser.add_argument(
        '--period', type=int, default=3, help='Previous 3 days.',
    )
    parser.add_argument(
        '--period_width', type=int, default=2, 
        help='2 hours before and after of predicted time.',
    )
    parser.add_argument(
        '--horizon', type=int, default=1, help='Predcition horizon.'
    )
    args = parser.parse_args()
    main(args)