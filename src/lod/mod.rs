use bevy::{
    color::palettes::{css::RED, tailwind::RED_500},
    ecs::{component::Component, system::Commands},
    math::{
        Dir3, Vec2, Vec3,
        bounding::{Aabb3d, BoundingSphere, IntersectsVolume},
        primitives::Cuboid,
    },
    prelude::*,
    render::mesh::{Mesh, Mesh3d, MeshBuilder, PlaneMeshBuilder},
};
use kdtree::KdTree;
use rand::Rng;

use crate::get_mesh_positions;

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

const MESH_SUBDIVISIONS: u32 = 9;
const TREE_DEPTH: usize = 5;
const MAP_SIZE: f32 = 256.0;
const RANGE_MIN_DIS: f32 = 20.0;

// type MeshGrid = [[f64; NODE_SIZE]; NODE_SIZE];

fn create_mesh_node(size: f32) -> Mesh {
    let normal = Dir3::new(Vec3::new(0.0, 1.0, 0.0)).unwrap();
    let size_vec = Vec2::new(size, size);
    let mesh = PlaneMeshBuilder::new(normal, size_vec)
        .subdivisions(MESH_SUBDIVISIONS)
        .build();
    mesh
}

#[derive(Component)]
pub struct PatchLabel;

pub fn render_lod(
    mut commands: Commands,
    mock_camera: Query<&Transform, With<MockCamera>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mesh_query: Query<Entity, With<PatchLabel>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Ok(transform) = mock_camera.single() else {
        return;
    };

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
    let boundry_rect = Rect::new(-MAP_SIZE, -MAP_SIZE, MAP_SIZE, MAP_SIZE);

    let mut node = MeshNode::new(
        Vec2::new(0.0, 0.0),
        TREE_DEPTH - 1,
        Vec3::new(MAP_SIZE / 2.0, 1.0, MAP_SIZE / 2.0),
    );

    let mut patches = Vec::new();
    select_lod(&node, &mut patches, TREE_DEPTH - 1, &bounding_spheres);

    // remove previous patches
    for entity in mesh_query.iter() {
        commands.entity(entity).despawn();
    }
    println!("NUM PATCHES: {}", patches.len());
    for patch in patches {
        let mesh = create_mesh_node(patch.size.x * 1.95);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(Color::srgb(0.0, 1.0 - patch.level as f32 * 0.3, 1.0))),
            Transform::from_xyz(patch.center.x, 10.0, patch.center.y),
            PatchLabel,
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
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(Color::from(RED_500))),
        Transform::from_xyz(0.0, 10.0, 0.0),
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

    let speed = 80.0;
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
