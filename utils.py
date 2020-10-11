# import numpy as np

# def conv2d(x, w, b, stride=1, pad=0):
    # N, C, H, W = x.shape 
    # num_filters, _, filter_height, filter_width = w.shape

    # # Check dimensions
    # assert (W + 2 * pad - filter_width) % stride == 0, "width does not work"
    # assert (H + 2 * pad - filter_height) % stride == 0, "height does not work"

    # # Create output
    # out_height = (H + 2 * pad - filter_height) // stride + 1
    # out_width = (W + 2 * pad - filter_width) // stride + 1
    # out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    # res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    # out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    # out = out.transpose(3, 0, 1, 2)

    # return out

# def batchnorm2d()

# def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # # First figure out what the size of the output should be
    # N, C, H, W = x_shape
    # assert (H + 2 * padding - field_height) % stride == 0
    # assert (W + 2 * padding - field_height) % stride == 0
    # out_height = (H + 2 * padding - field_height) / stride + 1
    # out_width = (W + 2 * padding - field_width) / stride + 1

    # i0 = np.repeat(np.arange(field_height), field_width)
    # i0 = np.tile(i0, C)
    # i1 = stride * np.repeat(np.arange(out_height), out_width)
    # j0 = np.tile(np.arange(field_width), field_height * C)
    # j1 = stride * np.tile(np.arange(out_width), out_height)
    # i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    # j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    # k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    # return (k, i, j)


# def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    # """ An implementation of im2col based on some fancy indexing """
    # # Zero-pad the input
    # p = padding
    # x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    # cols = x_padded[:, k, i, j]
    # C = x.shape[1]
    # cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    # return cols