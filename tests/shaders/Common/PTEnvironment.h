
const vec3 LightDirection = normalize(vec3(1,4,-1));
const vec3 LightIntensity = vec3(1,1,1) * 3.14;

vec3 SampleSkybox(vec3 dir){
    vec3 L = dir;
    vec3 BG_COLORS[5] =
	{
		vec3(0.1f, 0.05f, 0.01f), // GROUND DARKER BLUE
		vec3(0.01f, 0.05f, 0.2f), // HORIZON GROUND DARK BLUE
		vec3(0.8f, 0.9f, 1.0f), // HORIZON SKY WHITE
		vec3(0.1f, 0.3f, 1.0f),  // SKY LIGHT BLUE
		vec3(0.01f, 0.1f, 0.7f)  // SKY BLUE
	};
	float BG_DISTS[5] =
	{
		-1.0f,
		-0.1f,
		0.0f,
		0.4f,
		1.0f
	};
	vec3 col = BG_COLORS[0];
	col = mix(col, BG_COLORS[1], vec3(smoothstep(BG_DISTS[0], BG_DISTS[1], L.y)));
	col = mix(col, BG_COLORS[2], vec3(smoothstep(BG_DISTS[1], BG_DISTS[2], L.y)));
	col = mix(col, BG_COLORS[3], vec3(smoothstep(BG_DISTS[2], BG_DISTS[3], L.y)));
	col = mix(col, BG_COLORS[4], vec3(smoothstep(BG_DISTS[3], BG_DISTS[4], L.y)));
	return col;
}

vec3 SampleSkyboxWithSun(vec3 dir) {
	float sun_angle = 40 * pi / 180; // 0.5 degree, sun's angular diameter from Earth
	float cos_sun_angle = cos(sun_angle / 2);
	float sun_area = 2 * pi * (1 - cos_sun_angle);
	return dot(LightDirection, dir) > cos_sun_angle ? LightIntensity / sun_area : SampleSkybox(dir);
}