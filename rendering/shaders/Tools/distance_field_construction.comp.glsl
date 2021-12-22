#version 450
layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0, r32f) uniform image3D SrcGrid;
layout (binding = 1, r32f) uniform image3D DstGrid;

layout (binding = 2) uniform Parameters {
    vec3 direction;
    float tau_reference;
} parameters;

bool rayBoxIntersect(vec3 bMin, vec3 bMax, vec3 P, vec3 D, out float tMin, out float tMax) {
    // un-parallelize D
    D.x = abs(D).x <= 0.000001 ? 0.000001 : D.x;
    D.y = abs(D).y <= 0.000001 ? 0.000001 : D.y;
    D.z = abs(D).z <= 0.000001 ? 0.000001 : D.z;
    vec3 C_Min = (bMin - P)/D;
    vec3 C_Max = (bMax - P)/D;
	tMin = max(max(min(C_Min[0], C_Max[0]), min(C_Min[1], C_Max[1])), min(C_Min[2], C_Max[2]));
	tMin = max(0.0, tMin);
	tMax = min(min(max(C_Min[0], C_Max[0]), max(C_Min[1], C_Max[1])), max(C_Min[2], C_Max[2]));
	return tMax > tMin && tMax > 0;
}

vec3 g_box_minim;
vec3 g_box_maxim;
ivec3 g_dim;

float SampleGrid(vec3 p){
    ivec3 coordinate = ivec3( g_dim * (p - g_box_minim)/(g_box_maxim - g_box_minim) );
    if (any(lessThan(coordinate, ivec3(0))) || any(greaterThanEqual(coordinate, g_dim-ivec3(1))))
        return 0;
    return imageLoad(SrcGrid, coordinate).x;
}

void main(){
    int index = int(gl_GlobalInvocationID.x);
    // Compute normalized grid box
    g_dim = imageSize(DstGrid);
    float maxDim = max(g_dim.x, max(g_dim.y, g_dim.z));
    g_box_maxim = g_dim * 0.5 / maxDim;
    g_box_minim = -g_box_maxim;
    // Compute current coordinate
    int vx = index % g_dim.x;
    int vy = (index / g_dim.x) % g_dim.y;
    int vz = index / (g_dim.x * g_dim.y);
    if (vz >= g_dim.z)
    return;// kernel coordinate outside volume
    vec3 x = g_box_minim + vec3(vx+0.5, vy+0.5, vz+0.5) * (g_box_maxim - g_box_minim)/g_dim;
    vec3 w = parameters.direction;
    // Compute box hit
    float tMin, tMax;
    if (!rayBoxIntersect(g_box_minim, g_box_maxim, x, w, tMin, tMax))
    return;// It should never happen
    x += w*tMin;
    float d = tMax - tMin;
    float fixed_step_size = 1.0 / maxDim; // assuming a normalization -0.5, 0.5
    // update min radius if necessary
    float minRadius = imageLoad(DstGrid, ivec3(vx, vy, vz)).x;

    float tau = 0;
    for (float rad = minRadius; rad < d; rad += fixed_step_size)
    {
        tau += fixed_step_size * SampleGrid(x + w * rad);
        if (tau > parameters.tau_reference)
        return;// no radius update.
    }

    minRadius -= fixed_step_size; // not consider minRadius sample twice.

    while (minRadius > 0)
    {
        tau += fixed_step_size * SampleGrid(x + w * minRadius);
        if (tau > parameters.tau_reference)
        break;
        minRadius -= fixed_step_size;
    }

    minRadius = max(minRadius, 0);
    imageStore(DstGrid, ivec3(vx, vy, vz), vec4(minRadius));
}
