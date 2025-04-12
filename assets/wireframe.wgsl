#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{ VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif


#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip
}


@group(2) @binding(100) var<storage, read> data: array<f32>;
@group(2) @binding(110) var<storage, read> normals: array<vec4f>;
@group(2) @binding(111) var<storage, read> tangents: array<vec4f>;
struct PatchState {
    level: u32,
    offset_x: f32,
    offset_y: f32,
    camera_pos: vec4f,
    ranges: array<vec4f, 16>,
    tree_depth: u32,
}
@group(2) @binding(101) var<uniform> patch_state: PatchState;
struct Vertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(4) tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef SKINNED
    @location(6) joint_indices: vec4<u32>,
    @location(7) joint_weights: vec4<f32>,
#endif
    @builtin(vertex_index) index: u32,
};


@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    var vertex = vertex_in;
    let map_width = 600;
    let map_height = 600u;
    let pl = patch_state.level;
    // let scale =pow(1.1, f32(pl));
    let map_center = vec2f(300.0, 300.0);
    // let x = vertex.position[0];
    // let z = vertex.position[2];    // let i = z*map_height + x;
    // let height = f32(patch_state.level)*5.0;

    // let a: f32 = textureLoad(texure, vec2u(x, z)).x;
    let offset = 600.0 / pow(2.0, f32(patch_state.level));
    let xi = vertex.position[0] + patch_state.offset_x - offset + 300.0;
    let zi = vertex.position[2] + patch_state.offset_y - offset + 300.0;
    let x = xi - 300.0;
    let z = zi - 300.0;

    let i = u32(600.0*min(round(zi - 0.0), 599.0) + min(round(xi - 0.0), 599.0));
    let height: f32 = data[i] + 0.02;

    let computed_normal = normals[i];
    let computed_tangent = tangents[i];

    
    vertex.position = vec3f(x, height, z);
    var out: VertexOutput;


    let mesh_world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    var world_from_local = mesh_world_from_local;

#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        // vertex.normal,
        computed_normal.xyz,
        vertex.instance_index
    );
#endif

#ifdef VERTEX_POSITIONS
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
#endif

#ifdef VERTEX_UVS_A
    out.uv = vec2f(x, z);
    // out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        // vertex.tangent,
        computed_tangent,
        // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
        // See https://github.com/gfx-rs/naga/issues/2416
        vertex.instance_index
    );
#endif

#ifdef VERTEX_COLORS
    // out.color = vertex.color;
    let camera_pos = patch_state.camera_pos.xyz;
    var vi = patch_state.tree_depth - patch_state.level;
    // if vi > 0 {
    //     vi -= 1;
    // }
    let range = patch_state.ranges[vi].x;
    let next_range = patch_state.ranges[i + 1].x;
    let rdis = next_range - range;
    let dis = distance(camera_pos, vertex.position) - range;
    let f = (1.0 - dis/rdis) - 0.8;

    out.color = vec4f(0.0, f, 0.0, 1.0);
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
    // See https://github.com/gfx-rs/naga/issues/2416
    out.instance_index = vertex.instance_index;
#endif

#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex.instance_index, mesh_world_from_local[3]);
#endif

    return out;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var out: FragmentOutput;
    let c = 0.0;
    out.color = vec4f(c, c, c, 1.0);
    out.color = in.color;
    return out;

}

