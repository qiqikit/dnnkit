// OpenCL kernel. Each work item takes care of one element of c
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void vector_add(
	__global float* a,
	__global float* b,
	__global float* c,
	const unsigned int n
)
{
	//work item functions
	//get_work_dim();

	//get_global_size(uint dim);
	//get_global_id(uint dim);
	//get_global_offset(uint dim);

	//get_local_size(uint dim);
	//get_local_id(uint dim);

	//get_num_groups(uint dim);
	//get_group_id(uint dim);

	//get global thread id
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);

    //get group id
    //const int group_id = get_group_id(0);
    //const int group_size = get_num_groups(0);

    //get local thread id
    //const int local_id = get_local_id(0);
    //const int local_size = get_local_size(0);

	for (int id=global_id; id<n; id+=global_size)
	{
	    c[id] = a[id] + b[id];
	}
}

__kernel void vector_mul(
	__global float* a,
	__global float* b,
	__global float* c,
	const unsigned int n
)
{
	//get global thread id
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);

	for (int id=global_id; id<n; id+=global_size)
	{
	    c[id] = a[id] * b[id];
	}
}

__kernel void vector_sum(
	__global float* a,
	__global float* s,
	__local float* l,
	const unsigned int n
)
{
	//get global thread id
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);

    //get group id
    const int group_id = get_group_id(0);
    const int group_size = get_num_groups(0);

    //get local thread id
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);

    l[local_id] = 0;
    for (int id=global_id; id<n; id+=global_size)
    {
        l[local_id] += a[id];
    }

    for (int stride=local_size/2; stride>0; stride/=2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < stride)
        {
            l[local_id] += l[local_id + stride];
        }
    }

    if (local_id == 0)
    {
        s[group_id] = l[local_id];
    }
}