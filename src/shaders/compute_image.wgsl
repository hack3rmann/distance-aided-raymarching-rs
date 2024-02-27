struct CallData {
    screen_width: u32,
    screen_height: u32,
    y_offset: u32,
    index: u32,
}

struct RenderConfiguration {
    to_light: vec3f,
    n_bounces: u32,
    camera_distance: f32,
    camera_phi: f32,
    camera_theta: f32,
    camera_vfov: f32,
    camera_target_pos: vec3f,
    max_n_raymarch_steps: u32,
    max_raymarch_distance: f32,
    raymarch_eps: f32,
    ao_len: f32,
    ao_n_steps: u32,
    ao_min_value: f32,
    ao_power: f32,
    normal_eps: f32,
    normal_lift: f32,
    ambient: f32,
    fresnel_power: f32,
    super_sample_angle: f32,
    _padding: u32,
}

@group(0)
@binding(3)
var<uniform> call_data: CallData;

@group(0)
@binding(2)
var<uniform> cfg: RenderConfiguration;

@group(0)
@binding(1)
var<storage, read> scene: array<u32>;

@group(0)
@binding(0)
var<storage, read_write> image: array<u32>;

fn shorten_color(color: vec3f) -> u32 {
    let c = clamp(color * 255.0, vec3f(0.0), vec3f(255.0));

    return u32(c.r) << 0 | u32(c.g) << 8 | u32(c.b) << 16 | u32(255) << 24;
}



/// -------------------- SDF section --------------------

fn ball_sdf(pos: vec3f, centre: vec3f, radius: f32) -> f32 {
    return length(pos - centre) - radius;
}

fn semispace_sdf(pos: vec3f, normal: vec3f, dist: f32) -> f32 {
    return dot(pos, normal) - dist;
}

fn straight_prism_sdf(pos: vec3f, offset: vec3f, sizes: vec3f) -> f32 {
    let q = abs(pos - offset) - sizes;
    return length(max(q, vec3f(0.0))) + min(0.0, max(q.x, max(q.y, q.z)));
}

fn torus_sdf(pos: vec3f, offset: vec3f, outer_radius: f32, inner_radius: f32) -> f32 {
    let q = vec2f(
        length((pos - offset).xz) - inner_radius,
        pos.y - offset.y,
    );

    return length(q) - outer_radius;
}

fn mandelbulb_sdf(pos: vec3f, offset: vec3f, n_steps: u32, power: f32) -> f32 {
    var z = pos - offset;
    var dr = 1.0;
    var r = 0.0;

    for (var i = 0u; i < n_steps; i++) {
        r = length(z);

        if r > 4.0 {
            break;
        }

        var theta = acos(z.z / r);
        var phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        let zr = pow(r, power);
        theta *= power;
        phi *= power;

        z = zr * vec3f(
            sin(theta) * cos(phi),
            sin(phi) * sin(theta),
            cos(theta),
        );

        z += pos - offset;
    }

    return 0.5 * log(r) * r / dr;
}

fn object_sdf(pos: vec3f, data_index: u32, obj_type: u32) -> f32 {
    let offset = vec3f(
        bitcast<f32>(scene[data_index + 3]),
        bitcast<f32>(scene[data_index + 4]),
        bitcast<f32>(scene[data_index + 5]),
    );

    switch obj_type {
        case 0u: {
            let radius = bitcast<f32>(scene[data_index + 6 + 3]);
            return ball_sdf(pos, offset, radius);
        }
        case 1u: {
            let sizes = vec3f(
                bitcast<f32>(scene[data_index + 6 + 1]),
                bitcast<f32>(scene[data_index + 6 + 2]),
                bitcast<f32>(scene[data_index + 6 + 3]),
            );

            return straight_prism_sdf(pos, offset, sizes);
        }
        case 2u: {
            let n_steps = scene[data_index + 6 + 2];
            let power = bitcast<f32>(scene[data_index + 6 + 3]);

            return mandelbulb_sdf(pos, offset, n_steps, power);
        }
        case 3u: {
            let normal = vec3f(
                bitcast<f32>(scene[data_index + 6 + 0]),
                bitcast<f32>(scene[data_index + 6 + 1]),
                bitcast<f32>(scene[data_index + 6 + 2]),
            );

            let distance = bitcast<f32>(scene[data_index + 6 + 3]);

            return semispace_sdf(pos - offset, normal, distance);
        }
        case 4u: {
            let inner_radius = bitcast<f32>(scene[data_index + 6 + 2]);
            let outer_radius = bitcast<f32>(scene[data_index + 6 + 3]);

            return torus_sdf(pos, offset, outer_radius, inner_radius);
        }
        default: {
            return 0.0;
        }
    }

    return 0.0;
}



/// -------------------- CSG operations section --------------------

fn sdf_union(lhs: f32, rhs: f32) -> f32 {
    return min(lhs, rhs);
}

fn sdf_smooth_union(lhs: f32, rhs: f32, param: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (rhs - lhs) / param, 0.0, 1.0);
    return mix(rhs, lhs, h) - param * h * (1.0 - h);
}

fn sdf_intersection(lhs: f32, rhs: f32) -> f32 {
    return max(lhs, rhs);
}

fn sdf_smooth_intersection(lhs: f32, rhs: f32, param: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (rhs - lhs) / param, 0.0, 1.0);
    return mix(rhs, lhs, h) + param * h * (1.0 - h);
}

fn sdf_difference(lhs: f32, rhs: f32) -> f32 {
    return sdf_intersection(lhs, sdf_compliment(rhs));
}

fn sdf_smooth_difference(lhs: f32, rhs: f32, param: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (rhs + lhs) / param, 0.0, 1.0);
    return mix(rhs, -lhs, h) + param * h * (1.0 - h);
}

fn sdf_compliment(distance: f32) -> f32 {
    return -distance;
}

fn sdf_generic_op(lhs: f32, rhs: f32, data_index: u32, op_type: u32) -> f32 {
    let param = bitcast<f32>(scene[data_index + 7 + 2]);

    switch op_type {
        case 1u: { return sdf_union(lhs, rhs); }
        case 2u: { return sdf_smooth_union(lhs, rhs, param); }
        case 3u: { return sdf_intersection(lhs, rhs); }
        case 4u: { return sdf_smooth_intersection(lhs, rhs, param); }
        case 5u: { return sdf_difference(lhs, rhs); }
        case 6u: { return sdf_smooth_difference(lhs, rhs, param); }
        case 7u: { return sdf_compliment(lhs); }
        default: { return 0.0; }
    }

    return 0.0;
}

fn object_color(data_index: u32) -> vec3f {
    return vec3f(
        bitcast<f32>(scene[data_index + 0]),
        bitcast<f32>(scene[data_index + 1]),
        bitcast<f32>(scene[data_index + 2]),
    );
}



/// -------------------- Distance algorithms section --------------------

fn distance_color_mix(
    left_color: vec3f, right_color: vec3f,
    left_distance: f32, right_distance: f32,
    op_type: u32,
) -> vec3f {
    switch op_type {
        case 5u, 6u: {
            let sum = -left_distance + right_distance;

            if abs(sum) > 0.01 {
                return (right_distance * left_color - left_distance * right_color) / sum;
            } else {
                return left_color;
            }
        }
        default: {
            return (right_distance * left_color + left_distance * right_color)
                / (left_distance + right_distance);
        }
    }
}

const STACK_SIZE = 15u;
const SCENE_ELEM_SIZE = 12u;

struct SdfResult {
    color: vec3f,
    distance: f32,
}

var<private> distance_stack: array<f32, STACK_SIZE>;
var<private> color_stack: array<vec3f, STACK_SIZE>;

fn sdf(pos: vec3f) -> SdfResult {
    var stack_len = 0u;

    for (var scene_len = arrayLength(&scene) / SCENE_ELEM_SIZE
        ; scene_len > 0u
        ; scene_len--
    ) {
        let scene_index = scene_len - 1u;
        let flags = scene[SCENE_ELEM_SIZE * scene_index + 1u];
        let data_index = SCENE_ELEM_SIZE * scene_index + 2u;
        let is_object = (flags & 7u) == 0u;

        if is_object {
            distance_stack[stack_len]
                = object_sdf(pos, data_index, 7u & (flags >> 3u));

            color_stack[stack_len] = object_color(data_index);

            stack_len++;
        } else {
            let op_type = flags & 7u;

            // CsgOp::Compliment is op_type = 7
            if op_type == 7u {
                // distance = distance_stack.pop()
                // distance_stack.push(distance.compliment())
                distance_stack[stack_len - 1]
                    = sdf_compliment(distance_stack[stack_len - 1]);
            } else {
                let left_distance = distance_stack[stack_len - 2];
                let right_distance = distance_stack[stack_len - 1];
                let left_color = color_stack[stack_len - 2];
                let right_color = color_stack[stack_len - 1];

                distance_stack[stack_len - 2] = sdf_generic_op(
                    left_distance, right_distance, data_index, op_type,
                );

                color_stack[stack_len - 2] = distance_color_mix(
                    left_color, right_color, left_distance, right_distance,
                    op_type,
                );

                stack_len--;
            }
        }
    }

    return SdfResult(color_stack[0], distance_stack[0]);
}

fn compute_normal(pos: vec3f) -> vec3f {
    let eps = vec2f(cfg.raymarch_eps, 0.0);

    return normalize(vec3f(
        (sdf(pos + eps.xyy).distance - sdf(pos - eps.xyy).distance) / cfg.raymarch_eps,
        (sdf(pos + eps.yxy).distance - sdf(pos - eps.yxy).distance) / cfg.raymarch_eps,
        (sdf(pos + eps.yyx).distance - sdf(pos - eps.yyx).distance) / cfg.raymarch_eps,
    ));
}

struct RaymarchHitInfo {
    n_steps: u32,
    distance: f32,
    pos: vec3f,
}

fn raymarch(ro: vec3f, rd: vec3f) -> RaymarchHitInfo {
    var pos = ro;
    var distance = 0.0;

    for (var i = 0u; i < cfg.max_n_raymarch_steps; i++) {
        let displacement = sdf(pos).distance;
        let next_pos = pos + rd * displacement;

        if displacement < cfg.raymarch_eps {
            return RaymarchHitInfo(i + 1, distance + displacement, next_pos);
        }

        pos = next_pos;
        distance += displacement;

        if distance > cfg.max_raymarch_distance {
            return RaymarchHitInfo(~0u, distance, next_pos);
        }
    }

    return RaymarchHitInfo(cfg.max_n_raymarch_steps, distance, pos);
}

fn compute_ambient_occlusion(pos: vec3f, normal: vec3f) -> f32 {
    let step_size = cfg.ao_len / f32(cfg.ao_n_steps);

    let unoccluded_value = step_size
        * f32(cfg.ao_n_steps * (cfg.ao_n_steps + 1) / 2);

    var occluded_value = 0.0;

    for (var i = 1u; i <= cfg.ao_n_steps; i++) {
        let distance = sdf(pos + f32(i) * step_size * normal).distance;
        occluded_value += max(0.0, distance);
    }

    return cfg.ao_min_value
        + (1.0 - cfg.ao_min_value)
        * pow(occluded_value / unoccluded_value, cfg.ao_power);
}

struct HitInfo {
    do_hit: bool,
    pos: vec3f,
    color: vec3f,
}

var<private> hit_color_stack: array<vec3f, 6>;
var<private> reflectance_stack: array<f32, 6>;

fn hit(origin: vec3f, direction: vec3f) -> HitInfo {
    var hit_pos = origin;
    let stack_len = cfg.n_bounces + 1;

    var ro = origin;
    var rd = direction;

    for (var i = 0u; i < cfg.n_bounces + 1; i++) {
        let hit = raymarch(ro, rd);

        if ~hit.n_steps == 0 {
            hit_color_stack[i] = sky_color(rd);
            reflectance_stack[i] = 0.0;
            continue;
        }

        let pos = hit.pos;

        if i == 0 {
            hit_pos = pos;
        }

        let normal = compute_normal(pos);

        let is_shadow = ~raymarch(
            pos + cfg.normal_lift * normal, cfg.to_light,
        ).n_steps != 0;

        let albedo = sdf(pos).color;
        var brightness: f32;

        if !is_shadow {
            brightness = max(cfg.ambient, dot(normal, cfg.to_light));
        } else {
            brightness = cfg.ambient;
        }

        let fresnel_factor = pow(1.0 - abs(dot(normal, rd)), cfg.fresnel_power);

        let ao = compute_ambient_occlusion(pos, normal);

        hit_color_stack[i] = ao * brightness * albedo;
        reflectance_stack[i] = fresnel_factor;

        ro = pos + cfg.normal_lift * normal;
        rd = reflect(rd, normal);
    }

    var color = hit_color_stack[stack_len - 1];

    for (var i = stack_len - 2; ~i != 0u; i--) {
        color = mix(hit_color_stack[i], color, reflectance_stack[i]);
    }

    return HitInfo(true, hit_pos, color);
}

fn sky_color(direction: vec3f) -> vec3f {
    let sunness = dot(direction, cfg.to_light);

    let mid_color = vec3f(0.61, 0.72, 0.82);
    let top_color = vec3f(0.2, 0.49, 0.75);

    let sky_mix = 1.0 - pow(1.0 - abs(direction.y), 10.0);
    let sky_color = mix(mid_color, top_color, sky_mix);
    let sun_param = 0.01;
    
    if sunness >= 1.0 - sun_param {
        let sun_color = vec3(1.0, 1.0, 0.88);
        let amount = (sunness + sun_param - 1.0) / sun_param;
        return mix(sky_color, sun_color, pow(amount, 1.0 - sun_param));
    }

    if 0.0 <= direction.y {
        return sky_color;
    } else {
        return mix(mid_color, vec3f(0.2), sky_mix);
    }
}



fn spherical_to_cartesian(radius: f32, theta: f32, phi: f32) -> vec3f {
    return radius * vec3f(
        sin(phi) * sin(theta),
        cos(phi),
        sin(phi) * cos(theta),
    );
}

fn rotate(src: vec3f, axis: vec3f, angle: f32) -> vec3f {
    return src * cos(angle)
        + cross(axis, src) * sin(angle)
        + axis * dot(axis, src) * (1.0 - cos(angle));
}

fn get_color(screen_coord: vec2f) -> vec3f {
    let aspect_ratio = f32(call_data.screen_height) / f32(call_data.screen_width);

    let camera_pos = cfg.camera_target_pos + spherical_to_cartesian(
        cfg.camera_distance, cfg.camera_theta, cfg.camera_phi,
    );
    let camera_direction = normalize(cfg.camera_target_pos - camera_pos);
    let camera_tangent = vec3f(cos(cfg.camera_theta), 0.0, -sin(cfg.camera_theta));
    let camera_bitangent = cross(camera_direction, camera_tangent);

    let fov_tan = tan(0.5 * cfg.camera_vfov);
    let ray_direction = normalize(camera_direction
        + (screen_coord.x / aspect_ratio) * fov_tan * camera_tangent
        + screen_coord.y * fov_tan * camera_bitangent
    );
    let ray_origin = camera_pos;

    let rotated_tangent = rotate(camera_tangent, ray_direction, cfg.super_sample_angle);
    let rotated_bitangent = rotate(camera_bitangent, ray_direction, cfg.super_sample_angle);

    let pixel_size = 2.0 / f32(call_data.screen_height);
    let offset_x = 0.25 * pixel_size * rotated_tangent;
    let offset_y = 0.25 * pixel_size * rotated_bitangent;

    var color = vec3f(0.0);
    color += hit(ray_origin, ray_direction + offset_x).color;
    color += hit(ray_origin, ray_direction - offset_x).color;
    color += hit(ray_origin, ray_direction + offset_y).color;
    color += hit(ray_origin, ray_direction - offset_y).color;

    return 0.25 * color;
}

const WORKGROUP_WIDTH = 8u;
const WORKGROUP_HEIGHT = 8u;

@compute
@workgroup_size(WORKGROUP_WIDTH, WORKGROUP_HEIGHT, 1)
fn compute_image(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let workgroup_index = workgroup_id.x
        + workgroup_id.y * num_workgroups.x
        + workgroup_id.z * num_workgroups.x * num_workgroups.y;

    let global_invocation_index
        = workgroup_index * WORKGROUP_WIDTH * WORKGROUP_HEIGHT
        + local_invocation_index;

    let screen_index = global_invocation_index
        + call_data.y_offset * call_data.screen_width
        + call_data.index * call_data.screen_width / 3;

    let x_screen = screen_index % call_data.screen_width;
    let y_screen = screen_index / call_data.screen_width;

    let screen_coord = vec2f(
        f32(2 * i32(x_screen) - i32(call_data.screen_width) + 1) / f32(call_data.screen_width),
        f32(2 * i32(y_screen) - i32(call_data.screen_height) + 1) / f32(call_data.screen_height),
    );

    image[global_invocation_index] = shorten_color(get_color(screen_coord));
}