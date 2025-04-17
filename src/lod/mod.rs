use std::f32::consts::PI;

use bevy::{
    asset::RenderAssetUsages,
    color::palettes::{
        css::{GREEN, RED, SALMON},
        tailwind::{BLUE_200, INDIGO_600, PURPLE_500, RED_300, RED_500, YELLOW_500},
    },
    ecs::{component::Component, system::Commands},
    image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor},
    math::{
        Dir3, Vec2, Vec3,
        bounding::{Aabb3d, BoundingSphere, BoundingVolume, IntersectsVolume},
        primitives::Cuboid,
    },
    pbr::{ExtendedMaterial, NotShadowCaster, TransmittedShadowReceiver},
    prelude::*,
    render::{
        mesh::{Indices, Mesh, Mesh3d, MeshBuilder, PlaneMeshBuilder, PrimitiveTopology},
        render_resource::ShaderType,
        view::NoFrustumCulling,
    },
};
use kdtree::KdTree;
use rand::{Rng, random_range, rng};

use crate::{
    CustomMaterial, EventTimer, HeightBuffer, MAP_HEIGHT, MAP_WIDTH, NormalBuffer, PATCH_SIZE,
    PatchState, RANGE_MIN_DIS, TREE_DEPTH, TangentBuffer, WireframeMaterial,
};

struct MeshNode2 {
    boundry: Aabb3d,
    level: usize,
    partial: bool,
}

impl MeshNode2 {
    fn new(boundry: Aabb3d, level: usize, partial: bool) -> Self {
        Self {
            boundry,
            level,
            partial,
        }
    }

    fn children(&self) -> Vec<MeshNode2> {
        let mut children = Vec::new();
        let cen = self.boundry.center();
        let min = self.boundry.min;
        let max = self.boundry.max;
        let half_size = (cen.x - min.x) / 2.0;
        let half_size = Vec3::new(half_size, 0.0, half_size);

        let min_x = (min.x + cen.x) / 2.0;
        let max_x = (max.x + cen.x) / 2.0;
        let min_y = (min.z + cen.z) / 2.0;
        let max_y = (max.z + cen.z) / 2.0;

        // let upper_left = Vec3::new(min_x, 0.0, max_y);
        // let upper_right = Vec3::new(max_x, 0.0, max_y);
        // let lower_left = Vec3::new(min_x, 0.0, min_y);
        // let lower_right = Vec3::new(max_x, 0.0, min_y);
        let upper_left = Vec3::new(min_x, 0.0, max_y);
        let upper_right = Vec3::new(max_x, 0.0, max_y);
        let lower_left = Vec3::new(min_x, 0.0, min_y);
        let lower_right = Vec3::new(max_x, 0.0, min_y);

        let upper_left_boundry = Aabb3d::new(upper_left, half_size);
        let upper_right_boundry = Aabb3d::new(upper_right, half_size);
        let lower_left_boundry = Aabb3d::new(lower_left, half_size);
        let lower_right_boundry = Aabb3d::new(lower_right, half_size);

        let mut child_level = self.level;
        if child_level != 0 {
            child_level -= 1;
        }

        children.push(MeshNode2::new(upper_left_boundry, child_level, false));
        children.push(MeshNode2::new(upper_right_boundry, child_level, false));
        children.push(MeshNode2::new(lower_left_boundry, child_level, false));
        children.push(MeshNode2::new(lower_right_boundry, child_level, false));

        children
    }
}

struct MeshNode {
    center: Vec2,
    level: usize,
    size: Vec2,
    boundry: Aabb3d,
    children: Vec<MeshNode>,
}

impl MeshNode {
    fn new(center: Vec2, level: usize, half_size: Vec3) -> MeshNode {
        let mut children = Vec::new();
        if level > 0 {
            let xquartersize = half_size.x / 2.0;
            let zquartersize = half_size.z / 2.0;
            children.push(MeshNode::new(
                center + Vec2::new(xquartersize, zquartersize),
                level - 1,
                half_size / 2.0,
            ));
            children.push(MeshNode::new(
                center + Vec2::new(-xquartersize, zquartersize),
                level - 1,
                half_size / 2.0,
            ));
            children.push(MeshNode::new(
                center + Vec2::new(xquartersize, -zquartersize),
                level - 1,
                half_size / 2.0,
            ));
            children.push(MeshNode::new(
                center + Vec2::new(-xquartersize, -zquartersize),
                level - 1,
                half_size / 2.0,
            ));
            // cc
        };
        let cube = Aabb3d::new(Vec3::new(center.x, 0.0, center.y), half_size);

        MeshNode {
            center,
            level,
            size: half_size.xz(),
            boundry: cube,
            children,
        }
    }
}

fn patch_coord2index(x: usize, z: usize, patch_size: usize) -> usize {
    z * patch_size + x
}

fn patch_index2coord(index: usize, patch_size: usize) -> (usize, usize) {
    let x = index % patch_size;
    let z = index / patch_size;
    (x, z)
}

fn create_patch_mesh(size: usize) -> Mesh {
    let resolution = size + 1;
    let mut positions: Vec<[f32; 3]> = vec![[0.0; 3]; resolution * resolution];
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..resolution * resolution {
        let (x, z) = patch_index2coord(i, resolution);
        positions[i as usize] = [x as f32, 5.0, z as f32];
        uvs.push([x as f32, z as f32]);
    }

    // create triangles
    // go throw each row of mesh, and create triangles for row
    // * ------ * -------
    // | \      |
    // |    \   |  etc...
    // |       \|
    // * ------ * -------
    for row in 0..(resolution - 1) {
        for col in 0..(resolution - 1) {
            let top_left = patch_coord2index(row + 1, col, resolution) as u32;
            let top_right = patch_coord2index(row + 1, col + 1, resolution) as u32;
            let bottom_left = patch_coord2index(row, col, resolution) as u32;
            let bottom_right = patch_coord2index(row, col + 1, resolution) as u32;
            let mut triangle1 = vec![top_left, bottom_left, bottom_right];
            // triangle1.reverse();
            let mut triangle2 = vec![top_left, bottom_right, top_right];
            // triangle2.reverse();
            indices.append(&mut triangle1);
            indices.append(&mut triangle2);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());

    let plen = positions.len();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0; 4]; plen]);
    mesh.compute_normals();
    mesh.generate_tangents().unwrap();

    mesh
}
#[derive(Component)]
pub struct PatchLabel(u32);

#[derive(Resource, Default)]
pub struct CdlodMaterials {
    materials: Vec<Handle<ExtendedMaterial<StandardMaterial, WireframeMaterial>>>,
}
#[derive(Resource, Default)]
pub struct EnableWireframe(bool);

pub fn render_lod(
    mut commands: Commands,
    mock_camera: Query<&Transform, With<MockCamera>>,
    asset_server: Res<AssetServer>,
    buffer: Res<HeightBuffer>,
    normal_buffer: Res<NormalBuffer>,
    tangent_buffer: Res<TangentBuffer>,
    input: Res<ButtonInput<KeyCode>>,
    mut cdlod_state: ResMut<CdlodMaterials>,
    mesh_query: Query<(Entity, &PatchLabel)>,
    time: Res<Time>,
    mut enable_wireframe: ResMut<EnableWireframe>,
    mut gizmos: Gizmos,
    mut timer: ResMut<EventTimer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut custom_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, CustomMaterial>>>,
    mut wire_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, WireframeMaterial>>>,
) {
    let Ok(transform) = mock_camera.single() else {
        return;
    };

    if input.just_pressed(KeyCode::KeyU) {
        enable_wireframe.0 = !enable_wireframe.0;
    }

    timer.field1.tick(time.delta());
    if !timer.field1.just_finished() {
        // return;
        // update camera pos and skip
        let p = transform.translation;
        let cp = Vec4::from((p, 1.0));
        for handle in &cdlod_state.materials {
            if let Some(mat) = wire_materials.get_mut(handle) {
                mat.extension.level.camera_pos = cp;
            }
        }
        return;
    }
    let mut rng = rng();
    let frame_id = rng.random_range(0_32..1000000);
    let count = mesh_query.iter().len();
    let mat_count = custom_materials.len();
    let wire_count = wire_materials.len();
    println!("ENTITY COUNT {}", count);
    println!("MAT COUNT {}", mat_count);
    println!("wire COUNT {}", wire_count);
    for (entity, label) in mesh_query.iter() {
        commands.entity(entity).despawn();
        // if label.0 != frame_id {
        //     // println!("DESPAWN {}", label.0);
        //     commands.entity(entity).despawn();
        // }
    }
    // if count != 0 {
    //     return;
    // }
    cdlod_state.materials.clear();

    let mut ranges = Vec::new();
    for i in 0..TREE_DEPTH {
        ranges.push(RANGE_MIN_DIS * 2.0_f32.powi(i as i32));
    }

    let mut bounding_spheres = Vec::new();
    let colors: Vec<Color> = vec![
        BLUE_200.into(),
        RED_300.into(),
        YELLOW_500.into(),
        GREEN.into(),
        PURPLE_500.into(),
    ];

    let mut ri = 0;
    // gizmos.rect(
    //     Isometry3d::new(Vec3::new(0.0, 2.0, 0.0), Quat::from_rotation_x(PI / 2.)),
    //     Vec2::new(MAP_WIDTH as f32, MAP_HEIGHT as f32),
    //     Color::WHITE,
    // );
    for r in &ranges {
        // let sphere = Sphere::new(r);
        let bsphere = BoundingSphere::new(transform.translation, *r);
        // gizmos.sphere(transform.translation, *r, colors[ri]);
        bounding_spheres.push(bsphere);
        ri += 1;
    }

    let camera_center = transform.translation.xz();
    println!("CAMERA {:?}", camera_center);
    let map_boundry = Aabb3d::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(MAP_WIDTH as f32 / 2.0, 0.0, MAP_HEIGHT as f32 / 2.0),
    );

    // Root node encompasses entire map
    let root_node = MeshNode2::new(map_boundry, TREE_DEPTH - 1, false);

    let mut patches = Vec::new();
    select_lod2(&root_node, TREE_DEPTH - 1, &bounding_spheres, &mut patches);

    // remove previous patches
    println!("NUM PATCHES: {}", patches.len());

    // let mut patch_meshes = Vec::new();

    // for level in 0..TREE_DEPTH {
    //     let mesh = create_terrain_mesh_node(level);
    //     let mesh_handle = meshes.add(mesh);
    //     patch_meshes.push(mesh_handle);
    // }

    // let hm_handle = texture_buffer.0.clone();
    // println!("SPAWN {frame_id}");
    PatchState::assert_uniform_compat();
    let patch_mesh = create_patch_mesh(PATCH_SIZE);
    let mesh_handle = meshes.add(patch_mesh);
    for patch in patches {
        // if patch.partial {
        //     if partial_ran {
        //         continue;
        //     } else {
        //         partial_ran = true;
        //     }
        // }
        let side_length = get_side_length(patch.level);
        // let partial_side_length = get_side_length(patch.level);
        // let patch_size = side_length * (8 * PARTIAL_PATCH_SIZE - 1) as f32;
        let patch_size = 0.0;
        if patch.partial {
            // side_length = 0.0;
            // patch_size = 0.0;
        }

        let jitter = random_range(0.0_f32..5.0);
        // let gcen = patch.boundry.min + Vec3A::new(jitter, 0.0, 0.0);
        let gcen = patch.boundry.min;
        let mut ssize = 1.0;
        if patch.partial {
            ssize = 5.0;
        }
        // gizmos.sphere(gcen, ssize, colors[patch.level]);
        // gizmos.rect(
        //     Isometry3d::new(patch.boundry.center(), Quat::from_rotation_x(PI / 2.)),
        //     patch.boundry.half_size().xz() * 2.0,
        //     SALMON,
        // );
        let partial_flag = if patch.partial { 1 } else { 0 };
        let partial_flag = 0;
        let patch_state = PatchState::new(
            // pl as u32,
            patch.level as u32,
            patch.boundry.min.x,
            patch.boundry.min.z,
            transform.translation.to_array(),
            &ranges,
            TREE_DEPTH as u32,
            side_length,
            patch_size,
            partial_flag,
        );
        let cust_mat = ExtendedMaterial {
            base: StandardMaterial {
                perceptual_roughness: 0.9,
                ..Default::default()
            },
            extension: CustomMaterial {
                heightmap: buffer.0.clone(),
                normals: normal_buffer.0.clone(),
                tangents: tangent_buffer.0.clone(),
                level: patch_state,
                color_texture: Some(asset_server.load_with_settings(
                    // "textures/grass01.png",
                    "textures/ground2.png",
                    |s: &mut _| {
                        *s = ImageLoaderSettings {
                            sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
                                // rewriting mode to repeat image,
                                address_mode_u: ImageAddressMode::Repeat,
                                address_mode_v: ImageAddressMode::Repeat,
                                ..default()
                            }),
                            ..default()
                        }
                    },
                )),
                color2_texture: Some(asset_server.load_with_settings(
                    // "textures/grass01.png",
                    "textures/ground2_normal.png",
                    |s: &mut _| {
                        *s = ImageLoaderSettings {
                            sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
                                // rewriting mode to repeat image,
                                address_mode_u: ImageAddressMode::Repeat,
                                address_mode_v: ImageAddressMode::Repeat,
                                ..default()
                            }),
                            ..default()
                        }
                    },
                )),
                mountain_texture: Some(asset_server.load_with_settings(
                    "textures/mountain.png",
                    |s: &mut _| {
                        *s = ImageLoaderSettings {
                            sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
                                // rewriting mode to repeat image,
                                address_mode_u: ImageAddressMode::Repeat,
                                address_mode_v: ImageAddressMode::Repeat,
                                ..default()
                            }),
                            ..default()
                        }
                    },
                )),
                mountain_normals: Some(asset_server.load_with_settings(
                    "textures/mountain_normals.png",
                    |s: &mut _| {
                        *s = ImageLoaderSettings {
                            sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
                                // rewriting mode to repeat image,
                                address_mode_u: ImageAddressMode::Repeat,
                                address_mode_v: ImageAddressMode::Repeat,
                                ..default()
                            }),
                            ..default()
                        }
                    },
                )),
            },
        };
        let wire_mat = ExtendedMaterial {
            base: StandardMaterial {
                perceptual_roughness: 0.8,
                alpha_mode: AlphaMode::Blend,
                ..Default::default()
            },
            extension: WireframeMaterial {
                heightmap: buffer.0.clone(),
                normals: normal_buffer.0.clone(),
                tangents: tangent_buffer.0.clone(),
                level: patch_state,
            },
        };

        let mat_handle = custom_materials.add(cust_mat);
        // let mesh_handle = patch_meshes[patch.level].clone();
        let wire_handle = wire_materials.add(wire_mat);
        cdlod_state.materials.push(wire_handle.clone());
        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(mat_handle.clone()),
            NoFrustumCulling,
            // Transform::from_xyz(patch.center.x / 2.0, 0.0, patch.center.y / 2.0),
            PatchLabel(frame_id),
        ));

        if enable_wireframe.0 {
            commands.spawn((
                Mesh3d(mesh_handle.clone()),
                MeshMaterial3d(wire_handle.clone()),
                NoFrustumCulling,
                // Transform::from_xyz(patch.center.x / 2.0, 0.0, patch.center.y / 2.0),
                PatchLabel(frame_id),
            ));
        }
    }
}

struct PatchInfo {
    size: Vec2,
    center: Vec2,
    level: usize,
}

impl PatchInfo {
    fn from_node(node: &MeshNode) -> Self {
        Self {
            center: node.center,
            size: node.size,
            level: node.level,
        }
    }
}

struct QuadTree {}
impl QuadTree {
    fn new(map_boundry: Aabb3d, level_boundries: Vec<BoundingSphere>) -> QuadTree {
        QuadTree {}
    }
}

struct Patch {
    boundry: Aabb3d,
    level: usize,
    partial: bool,
}

impl Patch {
    fn new(boundry: Aabb3d, level: usize, partial: bool) -> Self {
        Self {
            boundry,
            level,
            partial,
        }
    }
}

// false, not handled
// true, handled
fn select_lod2(
    node: &MeshNode2,
    level: usize,
    ranges: &Vec<BoundingSphere>,
    patches: &mut Vec<Patch>,
) -> bool {
    let lod_boundry = ranges[level];

    // If we are not within the range of the current level,
    // then skip node
    if !node.boundry.intersects(&lod_boundry) {
        return false;
    }

    // At this point, we are within the current LOD range
    // Always add the highest detail LOD (0)
    if level == 0 {
        for child in node.children() {
            patches.push(Patch::new(child.boundry, level, false));
        }
        return true;
    }

    // We are not at the highest detail,
    // If the next LOD range doesn't intersect this node, then add
    // the whole node
    let next_lod_boundry = ranges[level - 1];
    if !node.boundry.intersects(&next_lod_boundry) {
        for child in node.children() {
            patches.push(Patch::new(child.boundry, level, false));
        }
    }
    // The next LOD range DOES intersect this node
    // We need to check to see which child nodes
    // are within the next node
    else {
        for child in node.children() {
            if !select_lod2(&child, level - 1, ranges, patches) {
                // Child node not within next range, add part of current node
                patches.push(Patch::new(child.boundry, level, true)); // true -> partial node
            }
        }
    }

    true
}

fn get_side_length(level: usize) -> f32 {
    let side_length =
        MAP_WIDTH as f32 / 2.0_f32.powf((TREE_DEPTH - level - 1) as f32) / (PATCH_SIZE) as f32;
    side_length * 0.5
}

fn select_lod(
    node: &MeshNode,
    patches: &mut Vec<PatchInfo>,
    level: usize,
    ranges: &Vec<BoundingSphere>,
) -> bool {
    let bounding_sphere = ranges[level];

    // Skip nodes not in current lodrange
    if !node.boundry.intersects(&bounding_sphere) {
        return false;
    }

    // Always add LOD0 within range
    if level == 0 {
        patches.push(PatchInfo::from_node(&node));
        return true;
    }

    // If node is only in 1 LOD range
    let next_bounding_sphere = ranges[level - 1];
    if !node.boundry.intersects(&next_bounding_sphere) {
        patches.push(PatchInfo::from_node(&node));
        return true;
    }

    // Otherwise, do more selecting
    for child in &node.children {
        if !select_lod(child, patches, level - 1, ranges) {
            patches.push(PatchInfo::from_node(&child));
        }
    }
    return true;
}

#[derive(Component)]
pub struct MockCamera;

pub fn setup_mock_camera(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cscale = 10.0;
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(Color::from(INDIGO_600))),
        Transform::from_xyz(0.0, 20.0, 0.0).with_scale(Vec3::new(cscale, cscale, cscale)),
        MockCamera,
    ));

    // let mesh = create_mesh_node(MAP_SIZE as f32);
    // commands.spawn((
    //     Mesh3d(meshes.add(mesh)),
    //     MeshMaterial3d(materials.add(Color::WHITE)),
    // ));
}
pub fn move_mock_camera(
    input: Res<ButtonInput<KeyCode>>,
    mut player: Query<&mut Transform, With<MockCamera>>,
    time: Res<Time>,
) {
    let Ok(mut transform) = player.single_mut() else {
        return;
    };

    let speed = 200.0;
    let translation = transform.translation;

    if input.pressed(KeyCode::ArrowUp) {
        transform.translation.z -= speed * time.delta_secs();
    }
    if input.pressed(KeyCode::ArrowDown) {
        transform.translation.z += speed * time.delta_secs();
    }
    if input.pressed(KeyCode::ArrowLeft) {
        transform.translation.x -= speed * time.delta_secs();
    }
    if input.pressed(KeyCode::ArrowRight) {
        transform.translation.x += speed * time.delta_secs();
    }
}
