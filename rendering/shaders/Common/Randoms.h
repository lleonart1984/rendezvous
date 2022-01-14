#ifndef RANDOMS_H
#define RANDOMS_H

#include "Maths.h"

uint TausStep(uint z, int S1, int S2, int S3, uint M) { uint b = (((z << S1) ^ z) >> S2); return ((z & M) << S3) ^ b; }

uint LCGStep(uint z, uint A, uint C) { return A * z + C; }

float random(inout uvec4 rng_state)
{
	rng_state.x = TausStep(rng_state.x, 13, 19, 12, 4294967294);
	rng_state.y = TausStep(rng_state.y, 2, 25, 4, 4294967288);
	rng_state.z = TausStep(rng_state.z, 3, 11, 17, 4294967280);
	rng_state.w = LCGStep(rng_state.w, 1664525, 1013904223);
	return 2.3283064365387e-10 * (rng_state.x ^ rng_state.y ^ rng_state.z ^ rng_state.w);
}

uvec4 initializeRandom(uint seed) {
    // FNV HASH
    seed = seed ^ 0xF2178221;
    seed = seed * (2*3*5*7*11*13+1);
    uvec4 rng_state = uvec4(seed);
    // Advance some randoms to reduce correlation
    for (int i = 0; i < seed % 7 + 2; i++)
        random(rng_state);
    return rng_state;
}

void createOrthoBasis(vec3 N, out vec3 T, out vec3 B)
{
    float sign = N.z >= 0 ? 1 : -1;
    float a = -1.0f / (sign + N.z);
    float b = N.x * N.y * a;
    T = vec3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = vec3(b, sign + N.y * N.y * a, -N.y);
}

vec3 randomDirection(inout uvec4 rng_state, vec3 D) {
	float r1 = random(rng_state);
	float r2 = random(rng_state) * 2 - 1;
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	createOrthoBasis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 randomHSDirection(inout uvec4 rng_state, vec3 D) {
	float r1 = random(rng_state);
	float r2 = random(rng_state);
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	createOrthoBasis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

//void d_randomHSDirection(inout uvec4 rng_state, in vec3 N, out vec3 w_out, out float pdf, out mat3x3 dwout_N)
//{
//    float r1 = random(rng_state);
//	float r2 = random(rng_state);
//	float sqrR2 = r2 * r2;
//	float two_pi_by_r1 = two_pi * r1;
//	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
//	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
//	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
//	float z = r2;
//	vec3 t0, t1;
//	createOrthoBasis(N, t0, t1);
//	dwout_N = mat3x3(t0, t1, N); // TODO: or transpose?
//	pdf = 1.0 / two_pi;
//	w_out = t0 * x + t1 * y + N * z;
//}

vec3 randomHSDirectionCosineWeighted(inout uvec4 rng_state, vec3 N, out float NdotD)
{
	vec3 t0, t1;
	createOrthoBasis(N, t0, t1);

	while (true) {
		float x = random(rng_state) * 2 - 1;
		float y = random(rng_state) * 2 - 1;
		float d2 = x * x + y * y;
		if (d2 > 0.001 && d2 < 1)
		{
			float z = sqrt(1 - d2);
			NdotD = z;
			return t0 * x + t1 * y + N * z;
		}
	}
	return vec3(0,0,0);
}

#endif