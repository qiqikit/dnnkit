// OpenCL kernel. Each work item takes care of one element of c
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable

//float kernels
__kernel void pooling_argmax(
	__global float* input_array,    /* input_array */
	__global int* max_index,        /* max index out */
	const int4 input_shape,         /* input_shape: dim of input {batch_size, chanel_size, input_width, input_height} */
	const int4 input_param          /* input_param: {ksize_width, ksize_height, stride_horizontal, stride_vertical} */
)
{
	//get global thread id
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);

    const int batch_size = input_shape.x;
    const int chanel_size = input_shape.y;
    const int input_width = input_shape.z;
    const int input_height = input_shape.w;

    const int ksize_width = input_param.x;
    const int ksize_height = input_param.y;
    const int stride_h = input_param.z;
    const int stride_v = input_param.w;

    const int output_width = (int)((input_width + ksize_width - stride_h) / stride_h);
    const int output_height = (int)((input_height + ksize_height - stride_v) / stride_v);

    const int output_len = batch_size * chanel_size * output_width * output_height;

    //__private float roi_cache[100];

	for (int id=global_id; id<output_len; id+=global_size)
	{
        const int batch = id / (chanel_size * output_width * output_height);
        const int chanel = id % (chanel_size * output_width * output_height) / (output_width * output_height);
        const int output_row = id % (output_width * output_height) / output_height;
        const int output_col = id % output_height;

        int argmax =
                batch * chanel_size * input_width * input_height +
                chanel * input_width * input_height +
                output_row * stride_h * input_height +
                output_col * stride_v;

        float valmax = input_array[argmax];

        //const int block_width = output_row*stride_h+ksize_width;
        for (int row=output_row*stride_h; row<output_row*stride_h+ksize_width; row++)
        {
            //const int block_height = output_col*stride_v+ksize_height;
            for (int col=output_col*stride_v; col<output_col*stride_v+ksize_height; col++)
            {
                const int input_index =
                    batch * chanel_size * input_width * input_height +
                    chanel * input_width * input_height +
                    row * input_height +
                    col;

                if (input_array[input_index] > valmax)
                {
                    argmax = input_index;
                }
            }
        }

        max_index[id] = argmax;
	}
}
__kernel void pooling_valmax(
	__global float* input_array,    /* input_array */
	__global int* max_index,        /* max index out */
	__global float* max_value,      /* max value out */
	const unsigned int output_len   /* output_len */
)
{
	//get global thread id
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);

    //const int work_size = (output_len + global_size - 1) / global_size;
	for (int id=global_id; id<output_len; id+=global_size)
	{
        max_value[id] = input_array[max_index[id]];
	}
}
__kernel void pooling_clrdiff(
	__global float* diff,
	const unsigned int diff_len
)
{
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);

    for (int id=global_id; id<diff_len; id+=global_size)
	{
	    diff[id] = 0;
	}
}
__kernel void pooling_dmax(
	__global float* output_diff,    /* output diff */
	__global int* max_index,        /* max index */
	__global float* input_diff,     /* input diff */
	const int4 output_shape         /* dim of output_diff {batch_size, chanel_size, output_width, output_height} */
)
{
    //get global thread id
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);

    const int batch_size = output_shape.x;
    const int chanel_size = output_shape.y;
    const int output_width = output_shape.z;
    const int output_height = output_shape.w;

    const int output_len = batch_size * chanel_size * output_width * output_height;

    for (int id=global_id; id<output_len; id+=global_size)
	{
        const int input_index = max_index[id];
        input_diff[input_index] += output_diff[id];
	}
}
__kernel void pooling_dmax2(
    __global float* output_diff,   /* output diff */
    __global int* max_index,       /* max index */
	__global float* input_diff,    /* input diff */
	const int4 input_shape,        /* input shape: dim of input {batch_size, chanel_size, input_width, input_height} */
	const int4 input_param         /* input param: {ksize_width, ksize_height, stride_horizontal, stride_vertical} */
)
{
	//get global thread id
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);

    const int batch_size = input_shape.x;
    const int chanel_size = input_shape.y;

    const int input_width = input_shape.z;
    const int input_height = input_shape.w;
    const int input_len = batch_size * chanel_size * input_width * input_height;

    const int ksize_width = input_param.x;
    const int ksize_height = input_param.y;
    const int stride_h = input_param.z;
    const int stride_v = input_param.w;

    const int output_width = (int)((input_width + ksize_width - stride_h) / stride_h);
    const int output_height = (int)((input_height + ksize_height - stride_v) / stride_v);
    const int output_len = batch_size * chanel_size * output_width * output_height;

	for (int id=global_id; id<input_len; id+=global_size)
	{
        const int batch = id / (chanel_size * input_width * input_height);
        const int chanel = id % (chanel_size * input_width * input_height) / (input_width * input_height);
        const int input_row = id % (input_width * input_height) / input_height;
        const int input_col = id % input_height;

        int output_row_begin = (input_row - ksize_width + 1 + (stride_h - 1)) / stride_h;
        int output_col_begin = (input_col - ksize_height + 1 + (stride_v -1)) / stride_v;
        int output_row_end = input_row / stride_h;
        int output_col_end = input_col / stride_v;

        if (output_row_begin < 0)
        {
            output_row_begin = 0;
        }
        if (output_col_begin < 0)
        {
            output_col_begin = 0;
        }
        if (output_row_end > output_width)
        {
            output_row_end = output_width;
        }
        if (output_col_end > output_height)
        {
            output_col_end = output_height;
        }

        for (int row=output_row_begin; row<=output_row_end; row++)
        {
            for (int col=output_col_begin; col<=output_col_end; col++)
            {
                const int output_index =
                    batch * chanel_size * output_width * output_height +
                    chanel * output_width * output_height +
                    row * output_height +
                    col;
                if (id == max_index[output_index])
                {
                    input_diff[id] += output_diff[output_index];
                }
            }
        }
	}
}
