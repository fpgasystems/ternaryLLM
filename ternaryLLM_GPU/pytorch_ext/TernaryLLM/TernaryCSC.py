import torch
from torch import tensor

def cnt_nnz(weight: tensor):
    cnt_neg = 0
    cnt_pos = 0
    for y in range(weight.size(1)):
        for x in range(weight.size(0)):
            value = weight[x, y].item()
            if value == -1:
                cnt_neg += 1
            elif value == 1:
                cnt_pos += 1
    return cnt_neg, cnt_pos


def convert_to_ter_csc(weight: tensor):
    """_summary_

    Args:
        weight (tensor): (inners, columns)

    Returns:
        _type_: _description_
    """
    # weight = weight.detach()
    w_neg_row_ind = []
    w_neg_col_ptr = []
    w_pos_row_ind = []
    w_pos_col_ptr = []

    nb_neg = 0
    nb_pos = 0
    for y in range(weight.size(1)):
        w_neg_col_ptr.append(nb_neg)
        w_pos_col_ptr.append(nb_pos)
        for x in range(weight.size(0)):  
            value = weight[x, y]
            if value.item() == -1:
                w_neg_row_ind.append(x)
                nb_neg += 1
            elif value.item() == 1:
                w_pos_row_ind.append(x)
                nb_pos += 1
    w_neg_col_ptr.append(nb_neg)
    w_pos_col_ptr.append(nb_pos)

    return w_neg_row_ind, w_neg_col_ptr, w_pos_row_ind, w_pos_col_ptr

def convert_to_ter_tiled_csc(weight: tensor, TILE_K: int):
    """_summary_

    Args:
        weight (tensor): (inners, columns)

    Returns:
        _type_: _description_
    """
    w1_neg_row_indice, _, w1_pos_row_indice, _ = convert_to_ter_csc(weight)

    inners = weight.size(0)
    columns = weight.size(1)

    import math
    size_csc_col_offset_tiled = columns * math.ceil(1.*inners/TILE_K)*2
    w1_neg_col_offset_tiled = torch.empty(size_csc_col_offset_tiled, dtype=torch.int32)
    w1_pos_col_offset_tiled = torch.empty(size_csc_col_offset_tiled, dtype=torch.int32)

    # /* compress tiled col offset */
    nb_neg = 0
    nb_pos = 0
    tileNumK = math.ceil(1.*inners/TILE_K)
    col_offset_x = tileNumK*2  # // each tile needs 2 offset [start, end)
    # // compress column
    for y in range(columns): 
        # // compress tile
        for tileId in range(tileNumK):
            offset = tileId*TILE_K
            w1_neg_col_offset_tiled[y*col_offset_x + tileId*2 + 0] = nb_neg
            w1_pos_col_offset_tiled[y*col_offset_x + tileId*2 + 0] = nb_pos
            for x in range(TILE_K): 
                if (weight[tileId*TILE_K+x, y] == -1):
                    w1_neg_row_indice[nb_neg] -= offset
                    nb_neg += 1
                elif (weight[tileId*TILE_K+x, y] == 1):
                    w1_pos_row_indice[nb_pos] -= offset
                    nb_pos += 1
                
            
            w1_neg_col_offset_tiled[y*col_offset_x + tileId*2 + 1] = nb_neg
            w1_pos_col_offset_tiled[y*col_offset_x + tileId*2 + 1] = nb_pos
        
    
    return tensor(w1_neg_row_indice, dtype=torch.int32), w1_neg_col_offset_tiled, tensor(w1_pos_row_indice, dtype=torch.int32), w1_pos_col_offset_tiled


def convert_to_ter_tiled_mcsc(weight: tensor, TILE_K: int):
    w1_neg_row_indice, w1_neg_col_offset_tiled, w1_pos_row_indice, w1_pos_col_offset_tiled = convert_to_ter_tiled_csc(weight, TILE_K)

    inners = weight.size(0)
    columns = weight.size(1)
    w1_cnt_neg = w1_neg_row_indice.size(0)
    w1_cnt_pos = w1_pos_row_indice.size(0)

    import math
    size_mcsc_col_offset_tiled = columns*4*math.ceil(1.*inners/TILE_K)
    w1_cnt_nnz = w1_cnt_neg + w1_cnt_pos
    w1_merged_row_indice_tiled =  torch.empty(w1_cnt_nnz, dtype=torch.int32)
    w1_merged_col_offset_tiled = torch.empty(size_mcsc_col_offset_tiled, dtype=torch.int32)
    
    tileNumK = math.ceil(1.*inners/TILE_K);  
    col_offset_tiled_csc = tileNumK*2  # // each tile needs 2 offset [start, end) in tiled csc
    col_offset_tiled_mcsc = tileNumK*4 # // each tile needs 4 offset [pos_neg_start, pos_neg_end, common_end, sign] in tiled mcsc
    prev_cnt = 0
    prev_nnz = 0

    for y in range(columns):
        for tileId in range(tileNumK):
            tile_offset = tileId*TILE_K
            tile_start_id = y*col_offset_tiled_csc + 2*tileId

            tile_neg_start = w1_neg_col_offset_tiled[tile_start_id].item()
            tile_neg_end = w1_neg_col_offset_tiled[tile_start_id+1].item()
            tile_pos_start = w1_pos_col_offset_tiled[tile_start_id].item()
            tile_pos_end = w1_pos_col_offset_tiled[tile_start_id+1].item()

            neg: int = tile_neg_end - tile_neg_start
            pos: int = tile_pos_end - tile_pos_start
            flag_min_neg = (neg <= pos)
            min_neg_pos = neg if flag_min_neg else pos
            max_neg_pos = pos if flag_min_neg else neg

            # /* compress interleaved number */
            cnt_interleave = min_neg_pos*2
            for k in range(min_neg_pos):
                w1_merged_row_indice_tiled[prev_nnz + k*2 + 0] = w1_pos_row_indice[tile_pos_start + k]
                w1_merged_row_indice_tiled[prev_nnz + k*2 + 1] = w1_neg_row_indice[tile_neg_start + k]
            
            # /* compress common number */
            cnt_remain = max_neg_pos - min_neg_pos
            for k in range(min_neg_pos, max_neg_pos):
                if flag_min_neg:
                    w1_merged_row_indice_tiled[prev_nnz + k + min_neg_pos] = w1_pos_row_indice[tile_pos_start + k]
                else:
                    w1_merged_row_indice_tiled[prev_nnz + k + min_neg_pos] = w1_neg_row_indice[tile_neg_start + k]
            
            tile_start_id = y*col_offset_tiled_mcsc + 4*tileId
            w1_merged_col_offset_tiled[tile_start_id + 0] = prev_cnt
            w1_merged_col_offset_tiled[tile_start_id + 1] = prev_cnt + cnt_interleave
            w1_merged_col_offset_tiled[tile_start_id + 2] = prev_cnt + cnt_interleave + cnt_remain
            w1_merged_col_offset_tiled[tile_start_id + 3] = 1 if flag_min_neg else -1

            prev_cnt += cnt_interleave + cnt_remain
            prev_nnz += neg + pos
        
    return w1_merged_row_indice_tiled, w1_merged_col_offset_tiled


if __name__ == "__main__":
    w = torch.randint(-1, 2, (16, 16))  # Shape (3, 4)
    # w1_neg_row_indice, w1_neg_col_offset_tiled, w1_pos_row_indice, w1_pos_col_offset_tiled = convert_to_ter_tiled_csc(w, 8)
    # print(w)
    # print(w1_neg_row_indice)
    # print(w1_neg_col_offset_tiled)
    w1_merged_row_indice_tiled, w1_merged_col_offset_tiled = convert_to_ter_tiled_mcsc(w, 8)
    print(w)
    print(w1_merged_row_indice_tiled)
    print(w1_merged_col_offset_tiled)