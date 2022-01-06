

float ComputeFresnel(float NdotL, float ratio)
{
	float oneMinusRatio = 1 - ratio;
	float onePlusRatio = 1 + ratio;
	float divOneMinusByOnePlus = oneMinusRatio / onePlusRatio;
	float f = divOneMinusByOnePlus * divOneMinusByOnePlus;
	return (f + (1.0 - f) * pow((1.0 - NdotL), 5));
}

void SurfaceScatter(inout uvec4 rng_seed, inout vec3 P,
in vec3 V, in vec3 N, in Material material,
out vec3 direction, out vec3 brdf_cos, out float pdf
)
{
    float VdotN = dot(V, N);
    bool entering = VdotN > 0;
    VdotN = entering ? VdotN : -VdotN;
    vec3 fN = entering ? N : -N;

    // Compute scattering
    float xi = random(rng_seed);
    if (xi < material.Model.x)// Diffuse
    {
        float NdotL;
        direction = randomHSDirectionCosineWeighted(rng_seed, N, NdotL);
        brdf_cos = material.Diffuse;
        pdf = 1.0;
        P += fN*0.000001;
        return;
    }
    xi -= material.Model.x;
    if (xi < material.Model.y)// Glossy
    {
        float NdotL;
        direction = randomHSDirection(rng_seed, N);
        vec3 H = normalize(V + direction);
        brdf_cos = material.Specular * pow(max(0, dot(H, N)), material.SpecularPower)
        * (2 + material.SpecularPower) * max(0, dot(direction, N)) * 0.5;// normalizing blinn model
        pdf = 1.0;
        P += fN*0.000001;
        return;
    }
    xi -= material.Model.y;

    if (xi < material.Model.z)// mirror
    {
        direction = reflect(-V, N);
        brdf_cos = material.Specular;
        pdf = 1;
        P += fN*0.000001;
        return;
    }
    xi -= material.Model.z;

    // Fresnel TODO

    float eta = entering ? material.RefractionIndex : 1 / material.RefractionIndex;
    float F = ComputeFresnel(VdotN, eta);
    vec3 T = refract (-V, fN, eta);
    if (all(equal(T, vec3(0))))// total internal reflection
    F = 1;
    if (xi < material.Model.w*F) // reflection
    {
        direction = reflect(-V, fN);
        brdf_cos = material.Specular;
        pdf = 1;
        P += fN*0.000001;
    }
    else
    {
        direction = T;
        brdf_cos = material.Specular;
        pdf = 1;
        P -= fN*0.000001; // traspass surface
    }
}