#ifndef VOLUME_SCATTERING_H
#define VOLUME_SCATTERING_H

#include "Randoms.h"

vec3 SampleHG(inout uvec4 seed, vec3 D, float phase_g) {
	float phi = random(seed) * 2 * pi;
    float xi = random(seed);
    float g2 = phase_g * phase_g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
    float one_over_2g = 0.5 / phase_g;

	float t = one_minus_g2 / (1.0f - phase_g + 2.0f * phase_g * xi);
	float invertcdf = one_over_2g * (one_plus_g2 - t * t);
	float cosTheta = abs(phase_g) < 0.001 ? 2 * xi - 1 : invertcdf;
	float sinTheta = sqrt(max(0, 1.0f - cosTheta * cosTheta));
	vec3 t0, t1;
	createOrthoBasis(D, t0, t1);
    return sinTheta * sin(phi) * t0 + sinTheta * cos(phi) * t1 + cosTheta * D;
}

float EvalHG(vec3 w_in, vec3 w_out, float phase_g){
	if (abs(phase_g) < 0.001)
		return 0.25 / pi;
    float g2 = phase_g * phase_g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
	float cosTheta = dot(w_in, w_out);
	return 0.25 / pi * (one_minus_g2) / pow(one_plus_g2 - 2 * phase_g * cosTheta, 1.5);
}

#endif
