use bevy::{
    asset::RenderAssetUsages,
    color::palettes::{css::RED, tailwind::RED_500},
    ecs::{component::Component, system::Commands},
    image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor},
    math::{
        Dir3, Vec2, Vec3,
        bounding::{Aabb3d, BoundingSphere, IntersectsVolume},
        primitives::Cuboid,
    },
    pbr::ExtendedMaterial,
    prelude::*,
    render::mesh::{Indices, Mesh, Mesh3d, MeshBuilder, PlaneMeshBuilder, PrimitiveTopology},
};
use kdtree::KdTree;
use rand::{Rng, rng};

use crate::{
    CustomMaterial, EventTimer, HeightBuffer, HeightMapTexture, MAP_WIDTH, NormalBuffer,
    PatchState, RANGE_MIN_DIS, TREE_DEPTH, TangentBuffer, coord2index, get_mesh_positions,
    index2coord,
};

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

// type MeshGrid = [[f64; NODE_SIZE]; NODE_SIZE];

// fn create_mesh_node(size: f32) -> Mesh {
//     let normal = Dir3::new(Vec3::new(0.0, 1.0, 0.0)).unwrap();
//     let size_vec = Vec2::new(size, size);
//     let mesh = PlaneMeshBuilder::new(normal, size_vec)
//         .subdivisions(MESH_SUBDIVISIONS)
//         .build();
//     mesh
const PATCH_WIDTH: usize = 40;
const PATCH_HEIGHT: usize = 10;
fn patch_coord2index(x: usize, z: usize) -> usize {
    z * PATCH_HEIGHT + x
}

fn patch_index2coord(index: usize) -> (usize, usize) {
    let x = index % PATCH_HEIGHT;
    let z = index / PATCH_HEIGHT;
    (x, z)
}

// Mesh for smallest patch
fn create_terrain_mesh_node(level: usize) -> Mesh {
    let node_height = PATCH_HEIGHT;
    let node_width = node_height;
    let mut positions: Vec<[f32; 3]> = vec![[0.0; 3]; node_width * node_height];
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let side_length =
        MAP_WIDTH as f32 / 2.0_f32.powf((TREE_DEPTH - level - 1) as f32) / (node_width - 1) as f32;

    // let mut rng = rng();
    for i in 0..node_width * node_height {
        let (x, z) = patch_index2coord(i);
        let xoffset = x as f32 * side_length;
        let zoffset = z as f32 * side_length;
        positions[i as usize] = [x as f32 * side_length, 5.0, z as f32 * side_length];
        uvs.push([xoffset, zoffset]);
    }

    // create triangles
    // go throw each row of mesh, and create triangles for row
    // * ------ * -------
    // | \      |
    // |    \   |  etc...
    // |       \|
    // * ------ * -------
    for row in 0..(node_height - 1) {
        for col in 0..(node_width - 1) {
            let top_left = patch_coord2index(row + 1, col) as u32;
            let top_right = patch_coord2index(row + 1, col + 1) as u32;
            let bottom_left = patch_coord2index(row, col) as u32;
            let bottom_right = patch_coord2index(row, col + 1) as u32;
            let mut triangle1 = vec![top_left, bottom_left, bottom_right];
            // triangle1.reverse();
            let mut triangle2 = vec![top_left, bottom_right, top_right];
            // triangle2.reverse();
            indices.append(&mut triangle1);
            indices.append(&mut triangle2);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_normals();
    mesh.generate_tangents().unwrap();

    mesh
}
fn create_mesh_node(size: f32) -> Mesh {
    let mut positions = Vec::new();

    positions.push([-1.0, 0.0, -1.0]);
    positions.push([1.0, 0.0, -1.0]);
    positions.push([-1.0, 0.0, 1.0]);
    positions.push([1.0, 0.0, 1.0]);
    for p in positions.iter_mut() {
        p[0] *= size / 2.0;
        p[2] *= size / 2.0;
    }
    let indices = vec![0, 2, 1, 1, 2, 3];
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_indices(Indices::U32(indices));

    mesh
}

#[derive(Component)]
pub struct PatchLabel(u32);

pub fn render_lod(
    mut commands: Commands,
    mock_camera: Query<&Transform, With<MockCamera>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,

    buffer: Res<HeightBuffer>,
    normal_buffer: Res<NormalBuffer>,
    tangent_buffer: Res<TangentBuffer>,
    // texture_buffer: Res<HeightMapTexture>,
    mesh_query: Query<(Entity, &PatchLabel)>,
    time: Res<Time>,
    mut timer: ResMut<EventTimer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut custom_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, CustomMaterial>>>,
) {
    let Ok(transform) = mock_camera.single() else {
        return;
    };

    timer.field1.tick(time.delta());
    if !timer.field1.just_finished() {
        return;
    }
    let mut rng = rng();
    let frame_id = rng.random_range(0_32..1000000);
    for (entity, label) in mesh_query.iter() {
        if label.0 != frame_id {
            // println!("DESPAWN {}", label.0);
            commands.entity(entity).despawn();
        }
    }

    let mut ranges = Vec::new();
    for i in 0..TREE_DEPTH {
        ranges.push(RANGE_MIN_DIS * 2.0_f32.powi(i as i32));
    }

    let mut bounding_spheres = Vec::new();

    for r in ranges {
        // let sphere = Sphere::new(r);
        let bsphere = BoundingSphere::new(transform.translation, r);
        bounding_spheres.push(bsphere);
    }

    let camera_center = transform.translation.xz();
    println!("CAMERA {:?}", camera_center);
    let bwidth = MAP_WIDTH as f32;
    // let boundry_rect = Rect::new(-bwidth, -bwidth, bwidth, bwidth);

    let node = MeshNode::new(
        Vec2::new(0.0, 0.0),
        TREE_DEPTH - 1,
        Vec3::new(bwidth / 2.0, 1.0, bwidth / 2.0),
    );

    let mut patches = Vec::new();
    select_lod(&node, &mut patches, TREE_DEPTH - 1, &bounding_spheres);

    // remove previous patches
    // println!("NUM PATCHES: {}", patches.len());

    let mut patch_meshes = Vec::new();

    for level in 0..TREE_DEPTH {
        let mesh = create_terrain_mesh_node(level);
        let mesh_handle = meshes.add(mesh);
        patch_meshes.push(mesh_handle);
    }

    // let hm_handle = texture_buffer.0.clone();
    // println!("SPAWN {frame_id}");
    for patch in patches {
        let pl = (TREE_DEPTH as i32 - patch.level as i32).max(0);
        let cust_mat = ExtendedMaterial {
            base: StandardMaterial::default(),
            extension: CustomMaterial {
                heightmap: buffer.0.clone(),
                normals: normal_buffer.0.clone(),
                tangents: tangent_buffer.0.clone(),
                level: PatchState::new(pl as u32, patch.center.x, patch.center.y),
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
        let w = MAP_WIDTH as f32 / 2.0_f32.powf((TREE_DEPTH - patch.level - 1) as f32);
        // println!(
        //     "PATCH LEVEL {}, CENTER {}, width: {w}",
        //     patch.level, patch.center
        // );
        let mat_handle = custom_materials.add(cust_mat);
        // let scale = 2.0_f32.powf(patch.level as f32 + 1.0) - 1.0;
        // let scale = 0.86 / 2.1 * (2.0_f32.powf(patch.level as f32));
        let scale = 1.0;
        let mesh_handle = patch_meshes[patch.level].clone();
        commands.spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(mat_handle.clone()),
            // Transform::from_xyz(patch.center.x / 2.0, 0.0, patch.center.y / 2.0),
            PatchLabel(frame_id),
        ));
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
        MeshMaterial3d(materials.add(Color::from(RED_500))),
        Transform::from_xyz(0.0, 10.0, 0.0).with_scale(Vec3::new(cscale, cscale, cscale)),
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

    let speed = 150.0;
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
