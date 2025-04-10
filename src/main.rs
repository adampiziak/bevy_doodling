use bevy::{
    asset::RenderAssetUsages,
    color::palettes::css::{BLUE, WHITE},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin},
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        extract_component::ExtractComponent,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        mesh::{Indices, PlaneMeshBuilder, VertexAttributeValues},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{storage_buffer, texture_storage_2d},
            *,
        },
        renderer::{RenderContext, RenderDevice},
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
        texture::GpuImage,
    },
    text::FontSmoothing,
};
use lod::{move_mock_camera, render_lod, setup_mock_camera};
use rand::{Rng, distr::uniform, rng};

const COMPUTE_SHADER_ASSET_PATH: &str = "compute.wgsl";
const TERRAIN_SHADER_PATH: &str = "terrain.wgsl";
const BUFFER_LEN: usize = 16;

mod lod;
#[derive(Resource)]
pub struct EventTimer {
    pub field1: Timer,
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    present_mode: bevy::window::PresentMode::AutoNoVsync,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            FpsOverlayPlugin {
                config: FpsOverlayConfig {
                    text_config: TextFont {
                        font_size: 42.0,
                        font: default(),
                        font_smoothing: FontSmoothing::default(),
                        ..default()
                    },
                    text_color: WHITE.into(),
                    refresh_interval: core::time::Duration::from_millis(70),
                    enabled: true,
                    ..Default::default()
                },
            },
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, CustomMaterial>>::default(),
            GpuReadbackPlugin,
            ExtractResourcePlugin::<HeightBuffer>::default(),
            ExtractResourcePlugin::<NormalBuffer>::default(),
            ExtractResourcePlugin::<TangentBuffer>::default(),
            ExtractResourcePlugin::<HeightMapTexture>::default(),
            // ExtractResourcePlugin::<ReadbackImage>::default(),
            ExtractResourcePlugin::<TerrainState>::default(),
        ))
        .insert_resource(EventTimer {
            // field1: Timer::from_seconds(1.0, TimerMode::Repeating),
            field1: Timer::from_seconds(0.05, TimerMode::Repeating),
        })
        .add_systems(Startup, setup)
        // .add_systems(Startup, setup_compute)
        .add_systems(Update, move_player)
        // .add_systems(Update, update_mesh)
        .add_systems(Update, compute_on_input)
        .add_systems(Startup, setup_camera)
        .add_systems(Startup, setup_mock_camera)
        .add_systems(Update, move_mock_camera)
        .add_systems(Update, render_lod)
        .run();
}

#[derive(Resource, Default, ExtractResource, Clone)]
struct TerrainState {
    stage: TerrainStage,
    buffer_size: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum TerrainStage {
    Idle,
    Start,
    Running,
    Reading,
    Finished,
}

impl Default for TerrainStage {
    fn default() -> Self {
        Self::Idle
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ComputeNodeLabel;

fn compute_on_input(
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
    mut terrain_state: ResMut<TerrainState>,
    query: Query<Option<&Readback>>,
) {
    // println!("TERRAIN STATE IS {:?}", terrain_state.stage);
    // if input.pressed(KeyCode::KeyJ) && terrain_state.stage == TerrainStage::Idle {
    //     terrain_state.stage = TerrainStage::Start;
    // }
    if terrain_state.stage == TerrainStage::Idle {
        terrain_state.stage = TerrainStage::Start;
    }
    // if terrain_state.stage == TerrainStage::Idle {
    //     for r in query {
    //         if let Some(ruw) = r {
    //             commands.entity(ruw);
    //         }
    //     }
    // }

    match terrain_state.stage {
        TerrainStage::Start => terrain_state.stage = TerrainStage::Running,
        TerrainStage::Running => terrain_state.stage = TerrainStage::Finished,
        TerrainStage::Finished => terrain_state.stage = TerrainStage::Idle,
        _ => {}
    }
}

#[derive(Resource)]
struct GpuBufferBindGroup(BindGroup);

// We need a plugin to organize all the systems and render node required for this example
struct GpuReadbackPlugin;
impl Plugin for GpuReadbackPlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ComputePipeline>().add_systems(
            Render,
            prepare_bind_group
                .in_set(RenderSet::PrepareBindGroups)
                // We don't need to recreate the bind group every frame
                .run_if(not(resource_exists::<GpuBufferBindGroup>)),
        );

        // Add the compute node as a top level node to the render graph
        // This means it will only execute once per frame
        render_app
            .world_mut()
            .resource_mut::<RenderGraph>()
            .add_node(ComputeNodeLabel, ComputeNode::default());
    }
}

/// The node that will execute the compute shader
#[derive(Default)]
struct ComputeNode {}
impl render_graph::Node for ComputeNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let Some(terrain_state) = world.get_resource::<TerrainState>() else {
            return Ok(());
        };

        if terrain_state.stage != TerrainStage::Running {
            return Ok(());
        }

        println!("COMPUTE");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ComputePipeline>();
        let bind_group = world.resource::<GpuBufferBindGroup>();

        if let Some(init_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("GPU readback compute pass"),
                        ..default()
                    });

            pass.set_bind_group(0, &bind_group.0, &[]);
            pass.set_pipeline(init_pipeline);
            let workgroup_x = (MAP_WIDTH + 7) / 8;
            let workgroup_z = (MAP_HEIGHT + 7) / 8;
            pass.dispatch_workgroups(workgroup_x as u32, 1, workgroup_z as u32);
        }
        Ok(())
    }
}

#[derive(Resource, ExtractResource, Clone)]
struct HeightBuffer(Handle<ShaderStorageBuffer>);
#[derive(Resource, ExtractResource, Clone)]
struct NormalBuffer(Handle<ShaderStorageBuffer>);
#[derive(Resource, ExtractResource, Clone)]
struct TangentBuffer(Handle<ShaderStorageBuffer>);
#[derive(Resource, ExtractResource, Clone)]
struct HeightMapTexture(Handle<Image>);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<ComputePipeline>,
    render_device: Res<RenderDevice>,
    buffer: Res<HeightBuffer>,
    normal_buffer: Res<NormalBuffer>,
    tangent_buffer: Res<TangentBuffer>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    // image: Res<HeightMapTexture>,
    // images: Res<RenderAssets<GpuImage>>,
) {
    let buffer = buffers.get(&buffer.0).unwrap();
    let tangent_buffer = buffers.get(&tangent_buffer.0).unwrap();
    let normal_buffer = buffers.get(&normal_buffer.0).unwrap();

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline.layout,
        &BindGroupEntries::sequential((
            buffer.buffer.as_entire_buffer_binding(),
            normal_buffer.buffer.as_entire_buffer_binding(),
            tangent_buffer.buffer.as_entire_buffer_binding(),
        )),
    );
    commands.insert_resource(GpuBufferBindGroup(bind_group));
}

#[derive(Resource)]
struct ComputePipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

#[derive(Component)]
struct Person;

fn update_mesh(
    mut commands: Commands,
    // storage_assets: Res<Assets<ShaderStorageBuffer>>,
    mut terrain_state: ResMut<TerrainState>,
    buffer: Res<HeightBuffer>,
    query: Query<Entity, With<Person>>, // buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
) {
    let count = query.iter().len();
    // println!("COUNT {count}");
    if terrain_state.stage == TerrainStage::Idle {
        for r in query.iter() {
            commands.entity(r).despawn();
        }
    }
    if terrain_state.stage == TerrainStage::Finished && count == 0 {
        terrain_state.stage = TerrainStage::Reading;
        println!("FINISH");
        let b = buffer.0.clone();
        let a = Readback::buffer(b);
        commands.spawn((a, Person)).observe(
            |trigger: Trigger<ReadbackComplete>,
             mesh_query: Query<&Mesh3d, With<Terrain>>,
             mut ecommands: Commands,
             mut terrain_state: ResMut<TerrainState>,
             mut meshes: ResMut<Assets<Mesh>>| {
                // This matches the type which was used to create the `ShaderStorageBuffer` above,
                // and is a convenient way to interpret the data.
                let data: Vec<[f32; 4]> = trigger.event().to_shader_type();
                for mesh_handle in mesh_query {
                    let mesh = meshes.get_mut(mesh_handle).unwrap();
                    if let Some(VertexAttributeValues::Float32x3(vals)) =
                        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
                    {
                        for (i, v) in vals.iter_mut().enumerate() {
                            v[1] = data[i][1];
                        }
                    }
                }
                // info!("Buffer {:?}", data);
                terrain_state.stage = TerrainStage::Idle;
                ecommands.entity(trigger.observer()).despawn();
            },
        );
    }
}

impl FromWorld for ComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        // let terrain_state = world.resource::<TerrainState>();
        let layout = render_device.create_bind_group_layout(
            None,
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<f32>>(false),
                    storage_buffer::<Vec<[f32; 4]>>(false),
                    storage_buffer::<Vec<[f32; 4]>>(false),
                ),
            ),
        );
        let shader = world.load_asset(COMPUTE_SHADER_ASSET_PATH);
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("GPU readback compute shader".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: Vec::new(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });
        ComputePipeline { layout, pipeline }
    }
}

pub fn get_mesh_positions<'a>(mesh: &'a Mesh) -> Option<&'a Vec<[f32; 3]>> {
    if let Some(VertexAttributeValues::Float32x3(vals)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        return Some(vals);
    } else {
        return None;
    }
}
// This struct defines the data that will be passed to your shader

#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType, Debug, Reflect)]
struct PatchState {
    level: u32,
    offset_x: f32,
    offset_y: f32,
}

impl PatchState {
    fn new(level: u32, offset_x: f32, offset_y: f32) -> Self {
        Self {
            level,
            offset_x,
            offset_y,
        }
    }
}

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct CustomMaterial {
    #[storage(100, read_only)]
    // #[texture(100)]
    // #[sampler(101)]
    heightmap: Handle<ShaderStorageBuffer>,
    #[uniform(101)]
    level: PatchState,
    #[texture(102)]
    #[sampler(103)]
    pub color_texture: Option<Handle<Image>>,
    #[texture(104)]
    #[sampler(105)]
    pub color2_texture: Option<Handle<Image>>,
    #[texture(106)]
    #[sampler(107)]
    pub mountain_texture: Option<Handle<Image>>,
    #[texture(108)]
    #[sampler(109)]
    pub mountain_normals: Option<Handle<Image>>,
    #[storage(110, read_only)]
    normals: Handle<ShaderStorageBuffer>,
    #[storage(111, read_only)]
    tangents: Handle<ShaderStorageBuffer>,
}
impl MaterialExtension for CustomMaterial {
    fn vertex_shader() -> ShaderRef {
        TERRAIN_SHADER_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        TERRAIN_SHADER_PATH.into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        TERRAIN_SHADER_PATH.into()
    }
}
const TREE_DEPTH: usize = 5;
const RANGE_MIN_DIS: f32 = 20.0;
const MAP_WIDTH: usize = 600;
const MAP_HEIGHT: usize = 600;

fn coord2index(x: usize, z: usize) -> usize {
    z * MAP_HEIGHT + x
}

fn index2coord(index: usize) -> (usize, usize) {
    let x = index % MAP_HEIGHT;
    let z = index / MAP_HEIGHT;
    (x, z)
}

fn create_terrain_mesh() -> Mesh {
    let mut positions: Vec<[f32; 3]> = vec![[0.0; 3]; MAP_HEIGHT * MAP_WIDTH];
    let mut indices: Vec<u32> = Vec::new();

    let mut rng = rng();
    for i in 0..MAP_HEIGHT * MAP_WIDTH {
        let (x, z) = index2coord(i);
        positions[i as usize] = [x as f32, rng.random_range(0_f32..10.0), z as f32];
    }

    // create triangles
    // go throw each row of mesh, and create triangles for row
    // * ------ * -------
    // | \      |
    // |    \   |  etc...
    // |       \|
    // * ------ * -------
    for row in 0..(MAP_HEIGHT - 1) {
        for col in 0..(MAP_WIDTH - 1) {
            let top_left = coord2index(row + 1, col) as u32;
            let top_right = coord2index(row + 1, col + 1) as u32;
            let bottom_left = coord2index(row, col) as u32;
            let bottom_right = coord2index(row, col + 1) as u32;
            let mut triangle1 = vec![top_left, bottom_left, bottom_right];
            let mut triangle2 = vec![top_left, bottom_right, top_right];
            indices.append(&mut triangle1);
            indices.append(&mut triangle2);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut custom_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, CustomMaterial>>>,
) {
    let terrain_state = TerrainState::default();
    let vertex_count = MAP_HEIGHT * MAP_WIDTH;
    let heightmap_buffer = vec![0.0; vertex_count];
    let normal_buffer = vec![[0.0; 4]; vertex_count];
    let tangent_buffer = vec![[0.0; 4]; vertex_count];
    let buffer = ShaderStorageBuffer::from(heightmap_buffer);
    let normal_buffer = ShaderStorageBuffer::from(normal_buffer);
    let tangent_buffer = ShaderStorageBuffer::from(tangent_buffer);
    let buffer = buffers.add(buffer);
    let normal_buffer = buffers.add(normal_buffer);
    let tangent_buffer = buffers.add(tangent_buffer);

    commands.insert_resource(terrain_state);
    commands.insert_resource(HeightBuffer(buffer));
    commands.insert_resource(NormalBuffer(normal_buffer));
    commands.insert_resource(TangentBuffer(tangent_buffer));
}

#[derive(Component)]
struct Terrain;

#[derive(Debug, Component)]
struct Player;

fn move_player(
    input: Res<ButtonInput<KeyCode>>,
    mut player: Query<&mut Transform, With<Player>>,
    time: Res<Time>,
) {
    let Ok(mut transform) = player.single_mut() else {
        return;
    };
    let translation = transform.translation;

    let speed = 80.0;
    if input.pressed(KeyCode::KeyW) {
        // transform.translation = Vec3 {
        //     z: translation.z - step,
        //     ..translation
        // };
        let forward = transform.rotation * -Vec3::Z;
        transform.translation += forward * speed * time.delta_secs();
    }
    let rotate_step = 0.01;
    if input.pressed(KeyCode::ShiftLeft) {
        transform.translation = Vec3 {
            y: translation.y - speed * time.delta_secs(),
            ..translation
        };
    }
    if input.pressed(KeyCode::Space) {
        transform.translation = Vec3 {
            y: translation.y + speed * time.delta_secs(),
            ..translation
        };
    }
    if input.pressed(KeyCode::KeyE) {
        // transform.rotate_x(rotate_step);
        transform.rotate_local_x(rotate_step * speed * time.delta_secs());
    }
    if input.pressed(KeyCode::KeyQ) {
        transform.rotate_local_x(-rotate_step * speed * time.delta_secs());
        // transform.rotate_x(-rotate_step);
    }
    if input.pressed(KeyCode::KeyZ) {
        transform.rotate_y(-rotate_step * speed * time.delta_secs());
    }
    if input.pressed(KeyCode::KeyX) {
        transform.rotate_y(rotate_step * speed * time.delta_secs());
        // transform.rotate_axis(axis, angle);
    }
    if input.pressed(KeyCode::KeyD) {
        // transform.translation = Vec3 {
        //     x: translation.x + step,
        //     ..translation
        // };
        let forward = transform.rotation * Vec3::X;
        transform.translation += forward * speed * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyA) {
        // transform.translation = Vec3 {
        //     x: translation.x - step,
        //     ..translation
        // };
        let forward = transform.rotation * Vec3::X;
        transform.translation -= forward * speed * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyS) {
        // transform.translation = Vec3 {
        //     z: translation.z + step,
        //     ..translation
        // };
        let forward = transform.rotation * -Vec3::Z;
        transform.translation -= forward * speed * time.delta_secs();
    }
}
fn setup_camera(mut commands: Commands) {
    commands
        // .spawn((
        //     Player,
        //     Transform::from_xyz(4., 700.0, 430.0),
        //     Visibility::default(),
        // ))
        .spawn((
            Player,
            Transform::from_xyz(4., 30.0, 100.0),
            Visibility::default(),
        ))
        .with_children(|parent| {
            // parent.spawn((WorldModelCamera,));

            // Spawn view model camera.
            parent.spawn((
                Camera3d::default(),
                Transform::from_xyz(10., 30., 10.).looking_to(
                    Vec3 {
                        x: 0.0,
                        y: -0.2,
                        z: -0.9,
                    },
                    Vec3::Y,
                ),
            ));
        });
    // commands.spawn((
    //     PointLight {
    //         intensity: 1000000.0,
    //         color: BLUE.into(),
    //         range: 100.0,
    //         ..Default::default()
    //     },
    //     Transform::from_xyz(0.0, 2.5, 0.0),
    // ));
    commands.spawn((
        DirectionalLight {
            illuminance: 6_000.0,

            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 300.0, 0.0).looking_to(
            Vec3 {
                x: -0.2,
                y: -0.12,
                z: 0.2,
            },
            Vec3::Y,
        ),
    ));

    // commands.spawn((
    //     DirectionalLight {
    //         illuminance: 16000.0,

    //         shadows_enabled: true,
    //         ..default()
    //     },
    //     Transform::from_xyz(0.0, 30.0, 0.0).looking_to(
    //         Vec3 {
    //             x: -0.2,
    //             y: -0.16,
    //             z: 0.2,
    //         },
    //         Vec3::Y,
    //     ),
    // ));
}
