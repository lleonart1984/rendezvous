#ifndef PRB_COMMON_H
#define PRB_COMMON_H

struct PRBRayHitPayload {
    // Tools (in out)
    uvec4 rng_seed;
    // Input, gradient to propagate in every scattering
    vec3 dL;
    vec3 L;
    // Output
    vec3 Position; // Scattered ray position
    vec3 Direction; // Scattered ray direction
    vec3 BRDF_cos; // Scattered albedo multiplied by cosine of theta
    float PDF; // Scattered ray pdf
};

#endif