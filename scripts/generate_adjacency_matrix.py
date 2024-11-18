import os 
import argparse
import h5py
import numpy as np 
import warnings
warnings.filterwarnings('ignore')

def get_neigh_nodes(x, y, row, col, selfloop=False):
    """
    Generate neighbours of the specified node `(x, y)` of a grid.
    """
    if((x < 0) or (x >= row) 
        or (y < 0) or (y >= col)):
        return []

    neigh_nodes = []
    for x_offset in range(-1, 2, 1):
        for y_offset in range(-1, 2, 1):
            if((not selfloop) 
                and (x_offset == 0) 
                and (y_offset == 0)):
                continue
            _x = x + x_offset
            _y = y + y_offset

            # filter invalid neighbour
            if((_x < 0) or (_x >= row) 
                or (_y < 0) or (_y >= col)):
                continue
            neigh_nodes.append(_x * col + _y)
    return neigh_nodes

def gen_adj_from_grid(row, col, output_file):
    num_nodes = row * col
    adj = np.zeros((num_nodes, num_nodes))

    for x in range(row):
        for y in range(col):
            nieghs = get_neigh_nodes(x, y, row, col)
            adj[x * col + y, nieghs] = 1.
    print('num_nodes: {}, num_edges: {}'.format(num_nodes, (adj > 0.).sum()))
    
    np.savez_compressed(
        output_file, 
        adj_mx=adj
    )

def gen_adjacency_matrix(dataset, output_dir):
    if dataset == 'BikeNYC':
        # h5py to npz
        data_path =  'data/BikeNYC/NYC14_M16x8_T60_NewEnd.h5'
        f = h5py.File(data_path)
        data = np.array(f['data'])
        seq_length, feat, row, col = data.shape
        gen_adj_from_grid(row, col, output_file=os.path.join(output_dir, dataset, 'adj_mx.npz'))
    elif dataset == 'TaxiBJ15':
        # h5py to npz
        data_path =  'data/TaxiBJ/BJ15_M32x32_T30_InOut.h5'
        f = h5py.File(data_path)
        data = np.array(f['data'])
        seq_length, feat, row, col = data.shape
        gen_adj_from_grid(row, col, output_file=os.path.join(output_dir, dataset, 'adj_mx.npz'))
    else:
        # newly added dataset can be implemented here.
        raise ValueError

def main(args):
    gen_adjacency_matrix(args.dataset, args.output_dir)

if __name__ == '__main__':
    '''
    python scripts/generate_adjacency_matrix.py --output_dir=data/ --dataset=BikeNYC

    python scripts/generate_adjacency_matrix.py --output_dir=data/ --dataset=TaxiBJ15
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='BikeNYC', help='Dataset signature.'
    )
    parser.add_argument(
        '--output_dir', type=str, default="data/", help="Output directory."
    )
    args = parser.parse_args()
    main(args)

