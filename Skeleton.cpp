#include "framework.h"

enum MaterialType { ROUGH, REFLECTIVE, REFRACTIVE };

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	float ior;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one)*(n - one) + kappa * kappa) / ((n + one)*(n + one) + kappa * kappa);
	}
};

struct RefractiveMaterial : Material {
	RefractiveMaterial(vec3 n) : Material(REFRACTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one)*(n - one)) / ((n + one)*(n + one));
		ior = n.x;
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center; radius = _radius; material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center)  / radius;
		hit.material = material;
		return hit;
	}
};

struct Paraboloid : public Intersectable {
	vec2 parameter;

	Paraboloid(const vec2& _parameter, Material* _material) {
		parameter = _parameter;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		hit.t = -1;

		// parametric equation calculation
		vec2 ray_start(ray.start.x, ray.start.y);
		vec2 ray_dir(ray.dir.x, ray.dir.y);

		float a = dot(parameter, ray_dir * ray_dir);
		float b = dot(parameter, 2 * ray_start * ray_dir) - ray.dir.z;
		float c = dot(parameter, ray_start * ray_start) - ray.start.z;
		// parametric equation calcuation

		float discr = (b * b - 4.0f * a * c);
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec3 comp = hit.position - vec3(0,0,0);

		// clipping for Paraboloid
		if(length(comp) > 2)
		{
			hit.t = -1;
		}
		// clipping for Paraboloid

		hit.normal = normalize(hit.position * vec3(2 * parameter.x, 2 * parameter.y, -1));
		hit.material = material;
		return hit;
	}
};

struct Plane : public Intersectable
{
	vec3 position;
	vec3 normal;
	float threshold;

	Plane(const vec3& _position, const vec3& _normal, Material * _material)
	{
		material = _material;
		position = _position;
		normal = _normal;
	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;
		vec3 dist = ray.start - position;

		hit.t = - (dot(normal, dist)) / dot(normal, ray.dir);

		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(vec3(normal.x, normal.y, normal.z));
		hit.material = material;

		return hit;
	}
};

struct Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye; lookat = _lookat; fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2 * (X + 0.5f) / windowWidth - 1) + up * (2 * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt)
	{
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, - d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye,lookat,up,fov);

	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;

public:
	void build() {
		vec3 eye = vec3(0, 10, 1), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(0, 0, -1), Le(1, 1, 1);
		lights.push_back(new Light(lightDirection, Le));


		vec3 kd1(0.3f, 0.2f, 0.1f);
		vec3 ks1(2, 2, 2);
		Material * material1 = new RoughMaterial(kd1, ks1, 50);

		vec3 kd2(0.3f, 0.5f, 0.3f);
		vec3 ks2(2, 2, 2);
		Material * material2 = new RoughMaterial(kd2, ks2, 50);

		vec3 kd3(0.1f, 0.1f, 0.1f);
		vec3 ks3(2, 2, 2);
		Material * material3 = new RoughMaterial(kd3, ks3, 50);

		vec3 kd4(0.3f, 0.3f, 0.5f);
		vec3 ks4(2, 2, 2);
		Material * material4 = new RoughMaterial(kd4, ks4, 50);

		vec3 kd5(0.5f, 0.4f, 0.3f);
		vec3 ks5(2, 2, 2);
		Material * material5 = new RoughMaterial(kd5, ks5, 50);

		objects.push_back(new Paraboloid(vec2(0.8, 0.8), material1));
		objects.push_back(new Sphere(vec3(0,0,3), 0.2f, material4));
		objects.push_back(new Plane(vec3(0,0,3.2), vec3(0,0,1), material5));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}


	vec3 trace(Ray ray, int depth = 0)
	{
	if (depth > 5) return La;
	Hit hit = firstIntersect(ray);
	if (hit.t < 0) return La;
	if (hit.material->type == ROUGH) {
		vec3 outRadiance = hit.material->ka * La;
		for (Light * light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
	float cosa = -dot(ray.dir, hit.normal);
	vec3 one(1,1,1);
	vec3 F = hit.material-> F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
	vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
	vec3 outRadiance = trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;

	if (hit.material -> type == REFRACTIVE)
	{
		float disc = 1 - (1 - cosa * cosa) / hit.material->ior / hit.material->ior; // scalar n
		if (disc >= 0)
		{
			vec3 refractedDir = ray.dir / hit.material->ior + hit.normal * (cosa / hit.material->ior - sqrt(disc));
			outRadiance = outRadiance + trace(Ray(hit.position - hit.normal * epsilon, refractedDir), depth + 1) * (one - F);
		}
	}
	return outRadiance;
}

	void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() { fragmentColor = texture(textureUnit, texcoord); }
)";

class FullScreenTexturedQuad {
	unsigned int vao = 0;	// vertex array object id
	unsigned int textureId = 0; // texture id
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight) {
		glGenVertexArrays(1, &vao);		// create 1 vertex array object
		glBindVertexArray(vao);				// make it active

		unsigned int vbo;				// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glGenTextures(1, &textureId);								// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);		// binding
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);	// binding
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);	// To GPU
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "outColor");		// create program for the GPU
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);													// Execute ray casting
	fullScreenTexturedQuad->LoadTexture(image);		// copy image to GPU as a texture
	fullScreenTexturedQuad->Draw();								// Display rendered image on screen
	glutSwapBuffers();														// exchange the two buffers
}

bool bool_animate = true;

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON))
	{
		bool_animate = false;
	}else {
		bool_animate = true;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	if (bool_animate) {
		scene.Animate(0.1f);
	}
	glutPostRedisplay();
}
