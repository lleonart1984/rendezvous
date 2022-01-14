struct d_float_float {
    float x;
    float dx_dp;
}

struct d_float_vec3 {
    float x;
    vec3 dx_dp;
}

struct d_vec3_vec3 {
    vec3 x;
    mat3 dx_dp;
}

d_float add (d_float_float a, d_float_float b){
    return d_float_float { a.x + b.x, a.dx_dp + b.dx_dp };
}

d_float sub (d_float_float a, d_float_float b){
    return d_float_float { a.x - b.x, a.dx_dp - b.dx_dp };
}

d_float mul (d_float_float a, d_float_float b){
    return d_float_float { a.x * b.x, a.dx_dp * b.x + b.dx_dp * a.x };
}

d_float div (d_float_float a, d_float_float b){
    return d_float_float { a.x / b.x, (a.dx_dp * b.x - b.dx_dp * a.x)/(b.x * b.x) };
}





