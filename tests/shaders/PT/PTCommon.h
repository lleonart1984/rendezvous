
struct RayHitPayload {
    // Tools (in out)
    uvec4 rng_seed;
    // Output
    vec3 Position; // Scattered ray position
    vec3 Direction; // Scattered ray direction
    vec3 BRDF_cos; // Scattered albedo multiplied by cosine of theta
    float PDF; // Scattered ray pdf
};

struct GeometryDesc {
    int StartVertex;
    int StartIndex;
    int TransformIndex;
};

struct InstanceDesc {
    int StartGeometry;
    int MaterialIndex;
};

struct Material {
    vec3 Diffuse;
    float Opacity;
    vec3 Specular;
    float SpecularPower;
    vec3 Emissive;
    float RefractionIndex;
    int DiffuseMap;
    int SpecularMap;
    int BumpMap;
    int MaskMap;
    vec4 Model;
};

struct Vertex {
    vec3 P;
    vec3 N;
    vec2 C;
    vec3 T;
    vec3 B;
};
