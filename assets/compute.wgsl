@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> normals: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tangents: array<vec4f>;
// @group(0) @binding(0) var texture: texture_storage_2d<r32float, read_write>;

@compute @workgroup_size(8, 1, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let map_height = 600u;
    let x = global_id.x;
    let z = global_id.z;
    let i = (z)*map_height + x;
    let pscale = 133.3;
    let perlin_x = f32(x)/pscale;
    let perlin_z = f32(z)/pscale;
    let perlin_pt = vec2f(perlin_x, perlin_z);
    let texel_size = vec2f(0.2, 0.2);
    data[i] += perlinNoise2(vec2f(perlin_x, perlin_z))/30.0;
    let normal = compute_normal(perlin_pt, texel_size);
    let tangent = compute_tangent(normal);
    normals[i] = vec4f(normal, 1.0);
    tangents[i] = vec4f(tangent, 1.0);
}

fn compute_normal(point: vec2f, texel_size: vec2f) -> vec3f {
    let left = point - vec2(texel_size.x, 0.0);
    let right = point + vec2(texel_size.x, 0.0);
    let down = point - vec2(0.0, texel_size.y);
    let up = point + vec2(0.0, texel_size.y);
    let hL = perlinNoise2(left);
    let hR = perlinNoise2(right);
    let hD = perlinNoise2(down);
    let hU = perlinNoise2(up);

    let dx = hL - hR;
    let dy = hD - hU;

    let normal = normalize(vec3(dx, 2.0, dy));
    return normal;
}

fn compute_tangent(normal: vec3<f32>) -> vec3<f32> {
    var up = vec3(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3(0.0, 0.0, 1.0); // fallback if normal is too close to Z+
    }
    let tangent = normalize(cross(up, normal));
    return tangent;
}


// MIT License. © Stefan Gustavson, Munrocket
//
fn permute4(x: vec4f) -> vec4f { return ((x * 34. + 1.) * x) % vec4f(289.); }
fn fade2(t: vec2f) -> vec2f { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2f) -> f32 {
    var Pi: vec4f = floor(P.xyxy) + vec4f(0., 0., 1., 1.);
    let Pf = fract(P.xyxy) - vec4f(0., 0., 1., 1.);
    Pi = Pi % vec4f(289.); // To avoid truncation effects in permutation
    let ix = Pi.xzxz;
    let iy = Pi.yyww;
    let fx = Pf.xzxz;
    let fy = Pf.yyww;
    let i = permute4(permute4(ix) + iy);
    var gx: vec4f = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
    let gy = abs(gx) - 0.5;
    let tx = floor(gx + 0.5);
    gx = gx - tx;
    var g00: vec2f = vec2f(gx.x, gy.x);
    var g10: vec2f = vec2f(gx.y, gy.y);
    var g01: vec2f = vec2f(gx.z, gy.z);
    var g11: vec2f = vec2f(gx.w, gy.w);
    let norm = 1.79284291400159 - 0.85373472095314 *
        vec4f(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 = g00 * norm.x;
    g01 = g01 * norm.y;
    g10 = g10 * norm.z;
    g11 = g11 * norm.w;
    let n00 = dot(g00, vec2f(fx.x, fy.x));
    let n10 = dot(g10, vec2f(fx.y, fy.y));
    let n01 = dot(g01, vec2f(fx.z, fy.z));
    let n11 = dot(g11, vec2f(fx.w, fy.w));
    let fade_xy = fade2(Pf.xy);
    let n_x = mix(vec2f(n00, n01), vec2f(n10, n11), vec2f(fade_xy.x));
    let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}
