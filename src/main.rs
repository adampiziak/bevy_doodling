use std::{f32::consts::PI, time::Duration};

use bevy::{
    asset::RenderAssetUsages,
    color::palettes::css::{FOREST_GREEN, WHITE},
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin},
    image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor},
    pbr::{
        CascadeShadowConfigBuilder, ExtendedMaterial, MaterialExtension, NotShadowCaster,
        NotShadowReceiver, light_consts::lux::FULL_DAYLIGHT, wireframe::WireframeConfig,
    },
    platform::collections::HashMap,
    prelude::*,
    render::{
        Render, RenderApp, RenderPlugin, RenderSet,
        extract_component::ExtractComponent,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        mesh::{ConeMeshBuilder, VertexAttributeValues},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{binding_types::storage_buffer, *},
        renderer::{RenderContext, RenderDevice},
        settings::{RenderCreation, WgpuSettings},
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
        view::VisibilityRange,
    },
    text::FontSmoothing,
};
use lod::{CdlodMaterials, EnableWireframe, move_mock_camera, render_lod, setup_mock_camera};
use rand::random_range;

const COMPUTE_SHADER_ASSET_PATH: &str = "compute.wgsl";
const TERRAIN_SHADER_PATH: &str = "terrain.wgsl";
const TERRAIN_PREPASS_PATH: &str = "terrain_prepass.wgsl";
const WIREFRAME_SHADER_PATH: &str = "wireframe.wgsl";

mod lod;
#[derive(Resource)]
pub struct EventTimer {
    pub field1: Timer,
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..Default::default()
                    }),

                    ..Default::default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        // WARN this is a native only feature. It will not work with webgl or webgpu
                        features: WgpuFeatures::POLYGON_MODE_LINE,
                        ..default()
                    }),
                    ..default()
                }),
            // You need to add this plugin to enable wireframe rendering
            // WireframePlugin::default(),
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
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, WireframeMaterial>>::default(),
            ExtractResourcePlugin::<ComputeManager>::default(),
            ComputePlugin,
            GpuReadbackPlugin,
            // ExtractResourcePlugin::<HeightBuffer>::default(),
            // ExtractResourcePlugin::<NormalBuffer>::default(),
            // ExtractResourcePlugin::<TangentBuffer>::default(),
            // ExtractResourcePlugin::<HeightMapTexture>::default(),
            ExtractResourcePlugin::<TerrainStageState>::default(),
        ))
        .add_event::<ComputeFinished>()
        .add_event::<RunComputePass>()
        .insert_resource(CdlodMaterials::default())
        .insert_resource(TerrainState::default())
        .insert_resource(EnableWireframe::default())
        .insert_resource(EventTimer {
            // field1: Timer::from_seconds(3.0, TimerMode::Repeating),
            // field1: Timer::from_seconds(2.0, TimerMode::Repeating),
            field1: Timer::from_seconds(0.3, TimerMode::Repeating),
            // field1: Timer::from_seconds(0.05, TimerMode::Repeating),
            // field1: Timer::from_seconds(0.02, TimerMode::Repeating),
        })
        .add_systems(Startup, (setup, setup_mock_camera, setup_camera))
        .add_systems(
            Update,
            (
                move_player,
                render_terrain,
                compute_on_input,
                coordinate_compute,
                move_mock_camera,
                render_lod,
            ),
        )
        .run();
}

#[derive(Resource, Default, ExtractResource, Clone)]
struct TerrainStageState {
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
    mut terrain_state: ResMut<TerrainStageState>,
    mut compute_event: EventWriter<RunComputePass>,
    query: Query<Option<&Readback>>,
) {
    // println!("TERRAIN STATE IS {:?}", terrain_state.stage);
    if input.pressed(KeyCode::KeyJ) && terrain_state.stage == TerrainStage::Idle {
        // terrain_state.stage = TerrainStage::Start;
        compute_event.write(RunComputePass);
    }
    // if terrain_state.stage == TerrainStage::Idle {
    //     terrain_state.stage = TerrainStage::Start;
    // }
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

    fn ready(&self, app: &App) -> bool {
        // let render_app = app.sub_app(RenderApp);
        println!("NOT READY");
        std::thread::sleep(Duration::from_secs(1));
        app.world().is_resource_added::<ComputeManager>()
    }

    fn finish(&self, app: &mut App) {
        let compute_manager = app.world().resource::<ComputeManager>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(compute_manager);
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

// Compute Terrain Uniform
struct ComputeTerrainData {
    mountain_lines: Vec<Vec2>,
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
        let Some(terrain_state) = world.get_resource::<TerrainStageState>() else {
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
            let workgroup_size = 16;
            let workgroup_x = (MAP_WIDTH + workgroup_size - 1) / workgroup_size;
            let workgroup_z = (MAP_HEIGHT + workgroup_size - 1) / workgroup_size;
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
    // buffer: Res<HeightBuffer>,
    // normal_buffer: Res<NormalBuffer>,
    // tangent_buffer: Res<TangentBuffer>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    // image: Res<HeightMapTexture>,
    // images: Res<RenderAssets<GpuImage>>,
    compute_manager: Res<ComputeManager>,
) {
    // let buffer = buffers.get(&buffer.0).unwrap();
    // let tangent_buffer = buffers.get(&tangent_buffer.0).unwrap();
    // let normal_buffer = buffers.get(&normal_buffer.0).unwrap();
    let mut binding_array = Vec::new();

    for (_, b) in &compute_manager.buffers {
        // let l = match b.data {
        //     ComputeBufferVector::F32(_) => storage_buffer::<Vec<f32>>(false),
        //     ComputeBufferVector::Vec4(_) => storage_buffer::<Vec<[f32; 4]>>(false),
        // };

        let handle = b.handle.clone().unwrap();
        let buffer = buffers.get(&handle).unwrap();

        binding_array.push(BindGroupEntry {
            binding: b.binding,
            resource: buffer.buffer.as_entire_binding(),
        });
    }

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline.layout,
        &binding_array, // &BindGroupEntries::sequential((
                        //     buffer.buffer.as_entire_buffer_binding(),
                        //     normal_buffer.buffer.as_entire_buffer_binding(),
                        //     tangent_buffer.buffer.as_entire_buffer_binding(),
                        // )),
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

#[derive(Resource, Default)]
struct TerrainState {
    heightmap: Vec<f32>,
}

fn create_tree(center: Vec3, index_offset: u32) -> (Vec<Vec3>, Vec<u32>, Vec<Vec2>) {
    let mut vertices = Vec::new();

    let mut indices = Vec::new();
    let mut uvs = Vec::new();
    // let tree_height = 1.6;
    let tree_height = random_range(0.8_f32..4.0) * 0.8;
    let tree_base_width = random_range(0.2_f32..0.8) * 1.5;
    // let tree_base_width = 0.5;
    // let tree_height = 1.6 * 10.0;
    // let tree_base_width = 0.5 * 10.0;
    let num_faces = 3;

    let face_offset = PI / num_faces as f32;
    let uv_warp = 10.0;

    let top_vertex = center + Vec3::new(0.0, tree_height, 0.0);
    uvs.push(Vec2::new(0.5, 1.0) / uv_warp);
    let top_index = index_offset;
    vertices.push(top_vertex);

    for i in 0..num_faces {
        let indices_start = vertices.len() as u32 + index_offset;
        let start_radians = face_offset * i as f32;
        let x1 = center.x + start_radians.cos() * tree_base_width;
        let x2 = center.x + (start_radians + PI).cos() * tree_base_width;
        let z1 = center.z + (start_radians).sin() * tree_base_width;
        let z2 = center.z + (start_radians + PI).sin() * tree_base_width;

        let vert1 = Vec3::new(x1, center.y, z1);
        let vert2 = Vec3::new(x2, center.y, z2);
        vertices.push(vert1);
        vertices.push(vert2);
        uvs.push(Vec2::new(1.0, 0.0) / uv_warp);
        uvs.push(Vec2::new(0.0, 0.0) / uv_warp);

        indices.push(top_index); // top vertex
        // if (x1 + z1 > x2 + z2) {
        indices.push(indices_start); // base right of triangle
        indices.push(indices_start + 1); // base left of triangle
        // } else {
        // indices.push(indices_start); // base right of triangle
        // indices.push(indices_start + 1); // base left of triangle
        // }
        // //double sided
        // indices.push(top_index); // top vertex
        // indices.push(indices_start + 1); // base left of triangle
        // indices.push(indices_start); // base right of triangle
    }

    //

    (vertices, indices, uvs)
}

fn render_terrain(
    mut compute_finished: EventReader<ComputeFinished>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    terrain_state: Res<TerrainState>,
    mut commands: Commands,
    query: Query<(Entity, &BoxLabel2)>,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !compute_finished.is_empty() {
        let chunk_divisions: usize = 8;

        let chunk_width: f32 = MAP_WIDTH as f32 / chunk_divisions as f32;
        let num_trees = 600_000 / (chunk_divisions.pow(2));

        for chunk_x in 0..chunk_divisions {
            for chunk_y in 0..chunk_divisions {
                let chunk_start_x = chunk_x as f32 * chunk_width;
                let chunk_start_y = chunk_y as f32 * chunk_width;
                for (entity, _) in query.iter() {
                    commands.entity(entity).despawn();
                }
                // for (entity, label) in box_query.iter() {
                //     ecommands.entity(entity).despawn();
                // }

                // let tree =
                //     asset_server.load(GltfAssetLabel::Scene(0).from_asset("tree/scene.gltf"));
                // let box_mesh = meshes.add(Cuboid::from_size(Vec3::splat(2.0)));
                // let box_mesh = meshes.add(ConeMeshBuilder::new(0.6, 1.5, 3).build());
                // let mut box_mat: StandardMaterial = Color::from(FOREST_GREEN).darker(0.16).into();
                // box_mat.perceptual_roughness = 1.0;
                // let box_mat = materials.add(box_mat);
                let data = &terrain_state.heightmap;
                // println!("COMPUT FINISHED");

                let mut forest_mesh =
                    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());

                let mut forest_vertices = Vec::new();
                let mut forest_indices = Vec::new();
                let mut forest_uvs = Vec::new();

                // let num_trees = 1_000_000;
                // let num_trees = 100_000;
                // let num_trees = 1;
                // let num_trees = 10;

                for _ in 0..num_trees {
                    let gap = 0.0;
                    let offset_x =
                        random_range((chunk_start_x + gap)..(chunk_start_x + chunk_width - gap));
                    let offset_z =
                        random_range((chunk_start_y + gap)..(chunk_start_y + chunk_width - gap));

                    if offset_x < 1.0
                        || offset_x > (MAP_WIDTH as f32 - 1.0)
                        || offset_z < 1.0
                        || offset_z > (MAP_HEIGHT as f32 - 1.0)
                    {
                        continue;
                    }

                    let i = offset_z.round() as usize * MAP_HEIGHT + offset_x as usize;
                    let i = i.min(data.len() - 1);
                    let height = data[i];
                    let roll = if height > 0.1 {
                        random_range(0.0_f32..height.powf(3.0))
                    } else {
                        0.05
                    };
                    if roll < 1.0 && height < 5.0 {
                        // println!("vertex count {}")
                        let (mut t_vers, mut t_inds, mut t_uvs) = create_tree(
                            Vec3::new(offset_x - 300.0, height, offset_z - 300.0),
                            forest_vertices.len() as u32,
                        );
                        forest_vertices.append(&mut t_vers);
                        forest_indices.append(&mut t_inds);
                        forest_uvs.append(&mut t_uvs);
                    }
                    //     let hlod_co = 500.0;
                    //     let hbias = 0.5;
                    //     commands.spawn((
                    //         // SceneRoot(tree.clone_weak()),
                    //         Mesh3d(box_mesh.clone()),
                    //         MeshMaterial3d(box_mat.clone()),
                    //         BoxLabel2,
                    //         NotShadowReceiver,
                    //         // NotShadowCaster,
                    //         VisibilityRange::abrupt(1.0, hlod_co),
                    //         // NotShadowCaster,
                    //         // Transform::from_xyz(0.0, 0.0, 0.0),
                    //         Transform::from_xyz(offset_x - 300.0, height + hbias, offset_z - 300.0),
                    //         // .with_scale(Vec3::splat(0.5)),
                    //     ));
                    //     commands.spawn((
                    //         // SceneRoot(tree.clone_weak()),
                    //         Mesh3d(box_mesh.clone()),
                    //         MeshMaterial3d(box_mat.clone()),
                    //         BoxLabel2,
                    //         NotShadowReceiver,
                    //         NotShadowCaster,
                    //         VisibilityRange::abrupt(hlod_co, 700.0),
                    //         // NotShadowCaster,
                    //         // Transform::from_xyz(0.0, 0.0, 0.0),
                    //         Transform::from_xyz(offset_x - 300.0, height + hbias, offset_z - 300.0),
                    //         // .with_scale(Vec3::splat(0.5)),
                    //     ));
                    // }
                }

                // let uvs = forest_vertices
                //     .iter()
                //     .map(|v| v.xz())
                //     .collect::<Vec<Vec2>>();

                forest_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, forest_vertices);
                forest_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, forest_uvs);
                forest_mesh.insert_indices(bevy::render::mesh::Indices::U32(forest_indices));

                forest_mesh.compute_normals();
                forest_mesh.generate_tangents().unwrap();
                let mesh = meshes.add(forest_mesh);
                let texture_handle = asset_server.load_with_settings("leaves.png", |s: &mut _| {
                    *s = ImageLoaderSettings {
                        sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
                            // rewriting mode to repeat image,
                            address_mode_u: ImageAddressMode::Repeat,
                            address_mode_v: ImageAddressMode::Repeat,
                            ..default()
                        }),
                        ..default()
                    }
                });
                let r = random_range(0.5_f32..1.1);
                let g = random_range(0.5_f32..1.1);
                let b = random_range(0.5_f32..1.1);
                let base = 0.55;
                let mat = StandardMaterial {
                    perceptual_roughness: 1.0,
                    base_color: Color::srgba(base, base, base + 0.12, 1.0),
                    reflectance: 0.05,
                    double_sided: true,
                    cull_mode: None,
                    // base_color: Color::srgba(0.0, , 0.0, 0.0),
                    base_color_texture: Some(texture_handle),
                    // thickness: 1.0,
                    // unlit: false,
                    ..Default::default()
                };
                // let mat = Color::WHITE;
                commands.spawn((Mesh3d(mesh), MeshMaterial3d(materials.add(mat)), BoxLabel2));
            }
        }
        compute_finished.clear();
    }
}

#[derive(Event)]
struct RunComputePass;

#[derive(Event)]
struct ComputeFinished;

fn coordinate_compute(
    mut commands: Commands,
    mut coordinator: ResMut<ComputeCoordinater>,
    mut terrain_state: ResMut<TerrainStageState>,
    time: Res<Time>,
    mut start_compute_event: EventReader<RunComputePass>,
) {
    // Waiting state, wait until elapsed time to read
    if let ComputeStage::Waiting(until) = coordinator.stage {
        if time.elapsed() > until {
            coordinator.stage = ComputeStage::Reading;
            commands.spawn(coordinator.readback()).observe(
                |trigger: Trigger<ReadbackComplete>,
                 mut ecommands: Commands,
                 mut terrain_state: ResMut<TerrainState>,
                 mut coordinator: ResMut<ComputeCoordinater>,
                 mut ev_compute_finished: EventWriter<ComputeFinished>| {
                    let data: Vec<f32> = trigger.event().to_shader_type();
                    terrain_state.heightmap = data;
                    ev_compute_finished.write(ComputeFinished);
                    ecommands.entity(trigger.observer()).despawn();
                    coordinator.stage = ComputeStage::Ready;
                },
            );
        }
        return;
    }

    // if compute is requested, then start
    if !start_compute_event.is_empty() {
        if coordinator.ready() {
            terrain_state.stage = TerrainStage::Start;
            coordinator.stage = ComputeStage::Waiting(time.elapsed() + Duration::from_millis(1000));
            start_compute_event.clear();
        }
    }
}

struct ComputePlugin;

impl Plugin for ComputePlugin {
    fn build(&self, app: &mut App) {
        let vertex_count = MAP_HEIGHT * MAP_WIDTH;
        let heightmap_buffer = vec![0.0; vertex_count];
        let normal_buffer = vec![[0.0; 4]; vertex_count];
        let tangent_buffer = vec![[0.0; 4]; vertex_count];
        let mut compute_manager = ComputeManager::default();
        compute_manager.add_buffer("HEIGHTMAP", ComputeBufferVector::F32(heightmap_buffer), 0);
        compute_manager.add_buffer("NORMAL", ComputeBufferVector::Vec4(normal_buffer), 1);
        compute_manager.add_buffer("TANGENT", ComputeBufferVector::Vec4(tangent_buffer), 2);
        // let mut buffer = ShaderStorageBuffer::from(heightmap_buffer);
        // buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
        // let normal_buffer = ShaderStorageBuffer::from(normal_buffer);
        // let tangent_buffer = ShaderStorageBuffer::from(tangent_buffer);
        // let buffer = buffers.add(buffer);
        // let normal_buffer = buffers.add(normal_buffer);
        // let tangent_buffer = buffers.add(tangent_buffer);

        // let coordinator = ComputeCoordinater::new(
        //     compute_manager
        //         .buffers
        //         .get("HEIGHTMAP")
        //         .unwrap()
        //         .handle
        //         .clone(),
        // );
        app.world_mut().insert_resource(compute_manager);
        println!("ADDED");
    }

    fn finish(&self, _app: &mut App) {}
}

#[derive(Resource, Default, ExtractResource, Clone)]
struct ComputeManager {
    pub buffers: HashMap<String, ComputeBuffer>,
}

impl ComputeManager {
    fn generate_handles(&mut self, buffers: &mut ResMut<Assets<ShaderStorageBuffer>>) {
        for (_, b) in self.buffers.iter_mut() {
            let mut buffer = match b.data.clone() {
                ComputeBufferVector::F32(data) => ShaderStorageBuffer::from(data),
                ComputeBufferVector::Vec4(data) => ShaderStorageBuffer::from(data),
            };
            buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
            b.handle = Some(buffers.add(buffer));
        }
    }
    fn add_buffer(&mut self, key: &str, data: ComputeBufferVector, binding: u32) {
        // let mut shaderbuffer = match data.clone() {
        //     ComputeBufferVector::F32(data) => ShaderStorageBuffer::from(data),
        //     ComputeBufferVector::Vec4(data) => ShaderStorageBuffer::from(data),
        // };

        // shaderbuffer.buffer_description.usage |= BufferUsages::COPY_SRC;
        // let handle = buffers.add(shaderbuffer);
        self.buffers.insert(
            key.into(),
            ComputeBuffer {
                binding,
                data,
                handle: None,
            },
        );
    }
}

#[derive(Clone)]
enum ComputeBufferVector {
    F32(Vec<f32>),
    Vec4(Vec<[f32; 4]>),
}

#[derive(Clone)]
pub struct ComputeBuffer {
    pub binding: u32,
    pub data: ComputeBufferVector,
    pub handle: Option<Handle<ShaderStorageBuffer>>,
}

// pub

impl FromWorld for ComputePipeline {
    fn from_world(world: &mut World) -> Self {
        // world.insert_resource(ComputeManager::default());
        let render_device = world.resource::<RenderDevice>();
        // let terrain_state = world.resource::<TerrainState>();
        let compute_manager = world.resource::<ComputeManager>();
        let mut layout_array = Vec::new();

        println!("---------");
        for (_, b) in &compute_manager.buffers {
            println!("BUFFER");
            let l = match b.data {
                ComputeBufferVector::F32(_) => storage_buffer::<Vec<f32>>(false),
                ComputeBufferVector::Vec4(_) => storage_buffer::<Vec<[f32; 4]>>(false),
            };
            layout_array.push((b.binding, l));
        }

        layout_array.sort_by_key(|t| t.0);
        let layouts: Vec<BindGroupLayoutEntry> = layout_array
            .into_iter()
            .map(|t| t.1.build(t.0, ShaderStages::COMPUTE))
            .collect();

        let layout = render_device.create_bind_group_layout(None, &layouts);
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
    camera_pos: Vec4,
    #[align(16)]
    ranges: [Vec4; 16],
    tree_depth: u32,
    side_length: f32,
    patch_size: f32,
    partial: u32,
}

impl PatchState {
    fn new(
        level: u32,
        offset_x: f32,
        offset_y: f32,
        camera_cen: [f32; 3],
        vec_ranges: &Vec<f32>,
        tree_depth: u32,
        side_length: f32,
        patch_size: f32,
        partial: u32,
    ) -> Self {
        let mut ranges = [Vec4::default(); 16];
        for (i, v) in vec_ranges.into_iter().enumerate() {
            ranges[i].x = *v;
        }
        Self {
            level,
            offset_x,
            offset_y,
            camera_pos: Vec4::from_array([camera_cen[0], camera_cen[1], camera_cen[2], 1.0]),
            ranges,
            tree_depth,
            side_length,
            patch_size,
            partial,
        }
    }
}
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct WireframeMaterial {
    #[storage(100, read_only)]
    // #[texture(100)]
    // #[sampler(101)]
    heightmap: Handle<ShaderStorageBuffer>,
    #[uniform(101)]
    level: PatchState,
    #[storage(110, read_only)]
    normals: Handle<ShaderStorageBuffer>,
    #[storage(111, read_only)]
    tangents: Handle<ShaderStorageBuffer>,
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
    fn prepass_vertex_shader() -> ShaderRef {
        TERRAIN_PREPASS_PATH.into()
    }

    // fn deferred_fragment_shader() -> ShaderRef {
    //     TERRAIN_SHADER_PATH.into()
    // }
}
impl MaterialExtension for WireframeMaterial {
    fn vertex_shader() -> ShaderRef {
        WIREFRAME_SHADER_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        WIREFRAME_SHADER_PATH.into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        WIREFRAME_SHADER_PATH.into()
    }
    fn specialize(
        pipeline: &bevy::pbr::MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &bevy::render::mesh::MeshVertexBufferLayoutRef,
        key: bevy::pbr::MaterialExtensionKey<Self>,
    ) -> std::result::Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.polygon_mode = PolygonMode::Line;
        Ok(())
    }
}
const TREE_DEPTH: usize = 3;
const RANGE_MIN_DIS: f32 = 200.0;
const MAP_WIDTH: usize = 600;
const MAP_HEIGHT: usize = 600;

const PATCH_SIZE: usize = 32;
#[derive(Component)]
pub struct BoxLabel2;

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut custom_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, CustomMaterial>>>,
    asset_server: Res<AssetServer>,
    mut compute_manager: ResMut<ComputeManager>,
) {
    let terrain_state = TerrainStageState::default();
    compute_manager.generate_handles(&mut buffers);
    let coordinater = ComputeCoordinater::new(
        compute_manager
            .buffers
            .get("HEIGHTMAP")
            .unwrap()
            .handle
            .clone()
            .unwrap(),
    );

    // // trees
    // // buffer.read
    commands.insert_resource(terrain_state);
    commands.insert_resource(coordinater);
    // commands.insert_resource(HeightBuffer(buffer));
    // commands.insert_resource(NormalBuffer(normal_buffer));
    // commands.insert_resource(TangentBuffer(tangent_buffer));
}

#[derive(PartialEq)]
enum ComputeStage {
    Ready,
    Waiting(Duration),
    Reading,
}

#[derive(Resource)]
struct ComputeCoordinater {
    stage: ComputeStage,
    heightmap: Vec<f32>,
    buffer: Handle<ShaderStorageBuffer>,
}

impl ComputeCoordinater {
    fn new(buffer: Handle<ShaderStorageBuffer>) -> ComputeCoordinater {
        return ComputeCoordinater {
            stage: ComputeStage::Ready,
            heightmap: vec![0.0; MAP_HEIGHT * MAP_WIDTH],
            buffer,
        };
    }

    fn ready(&self) -> bool {
        self.stage == ComputeStage::Ready
    }

    fn readback(&self) -> Readback {
        Readback::Buffer(self.buffer.clone())
    }

    fn update(&mut self, mut commands: Commands) {
        // self.stage = ComputeStage::Updating;
        /*
        let readback = Readback::buffer(self.buffer.clone());
        commands.spawn(readback).observe(
            |trigger: Trigger<ReadbackComplete>,
             mut ecommands: Commands,
             mut ev_compute_finished: EventWriter<ComputeFinished>| {
                let data: Vec<f32> = trigger.event().to_shader_type();
                ev_compute_finished.write(ComputeFinished(data));
                ecommands.entity(trigger.observer()).despawn();
            },
        );
                let sample = *data.get(50000).unwrap();
                if (sample).abs() > 1.0 {
                    println!("COMPUTE READBACK");
                    let rand_offset: f32 = 600.0;
                    // for (entity, label) in box_query.iter() {
                    //     ecommands.entity(entity).despawn();
                    // }

                    // let tree =
                    //     asset_server.load(GltfAssetLabel::Scene(0).from_asset("tree/scene.gltf"));
                    // let box_mesh = meshes.add(Cuboid::from_size(Vec3::splat(2.0)));
                    let box_mesh = meshes.add(ConeMeshBuilder::new(0.8, 1.5, 4).build());
                    let mut box_mat: StandardMaterial =
                        Color::from(FOREST_GREEN).darker(0.16).into();
                    box_mat.perceptual_roughness = 1.0;
                    let box_mat = materials.add(box_mat);
                    for _ in 0..20 {
                        // let offset_x = random_range(-rand_offset..rand_offset);
                        let offset_x = random_range(0_f32..rand_offset);
                        // let offset_y = random_range(-rand_offset..rand_offset);
                        // let offset_z = random_range(-rand_offset..rand_offset);
                        let offset_z = random_range(0_f32..rand_offset);

                        // buffer.ob

                        let i = offset_z.round() as usize * MAP_HEIGHT + offset_x as usize;
                        let i = i.min(data.len() - 1);
                        let height = data[i];
                        println!("{height}");
                        if height < 1.0 {
                            let hlod_co = 400.0;
                            ecommands.spawn((
                                // SceneRoot(tree.clone_weak()),
                                Mesh3d(box_mesh.clone()),
                                MeshMaterial3d(box_mat.clone()),
                                BoxLabel2,
                                NotShadowReceiver,
                                // NotShadowCaster,
                                VisibilityRange::abrupt(30.0, hlod_co),
                                // NotShadowCaster,
                                // Transform::from_xyz(0.0, 0.0, 0.0),
                                Transform::from_xyz(offset_x - 300.0, height, offset_z - 300.0),
                                // .with_scale(Vec3::splat(0.5)),
                            ));
                            ecommands.spawn((
                                // SceneRoot(tree.clone_weak()),
                                Mesh3d(box_mesh.clone()),
                                MeshMaterial3d(box_mat.clone()),
                                BoxLabel2,
                                NotShadowReceiver,
                                NotShadowCaster,
                                VisibilityRange::abrupt(hlod_co, 700.0),
                                // NotShadowCaster,
                                // Transform::from_xyz(0.0, 0.0, 0.0),
                                Transform::from_xyz(offset_x - 300.0, height, offset_z - 300.0),
                                // .with_scale(Vec3::splat(0.5)),
                            ));
                        }

                        //     ecommands.entity(trigger.observer()).despawn();
                    }
                } else {
                    println!("STILL WAITING");
                }
                // info!("Buffer {:?}", data);
                // terrain_state.stage = TerrainStage::Idle;
            },
        );
        */
    }
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

    let mut speed = 30.0;
    if input.pressed(KeyCode::ControlLeft) {
        speed *= 4.0;
    }
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
            NotShadowCaster,
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
    // commands.spawn(AmbientLight {
    //     brightness: 2000.0,
    //     ..Default::default()
    // });
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,

            shadows_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            first_cascade_far_bound: 400.0,

            maximum_distance: 1200.0,
            ..Default::default()
        }
        .build(),
        Transform::from_xyz(0.0, 300.0, 0.0).looking_to(
            Vec3 {
                x: -0.5,
                y: -0.15,
                z: 0.5,
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

fn toggle_wireframe(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<WireframeConfig>,
) {
    // Toggle showing a wireframe on all meshes
    if keyboard_input.just_pressed(KeyCode::KeyP) {
        config.global = !config.global;
    }
}
