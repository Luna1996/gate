use glam::{IVec3, Mat3, Vec3};

use gate_voxel::{CHUNK_SIZE, VolumeGrid, VolumeTransform, VoxelCoord};

use crate::classify::ContactVoxels;
use crate::field::{Field, NodeFill, world_aabb_of};

#[derive(Clone, Copy, Debug)]
pub struct MassProps {
  pub mass: f32,
  pub com: Vec3,
  pub inertia: Mat3,
}

pub fn mass_properties(grid: &VolumeGrid, density: f32) -> Option<MassProps> {
  let f = Field::new(grid);
  let mut n = 0u64;
  let mut sum = Vec3::ZERO;
  let mut moments = Mat3::ZERO;
  {
    let mut acc = |c: Vec3| {
      n += 1;
      sum += c;
      moments += Mat3::from_diagonal(Vec3::splat(c.length_squared() + 1.0 / 6.0))
        - Mat3::from_cols(c * c.x, c * c.y, c * c.z);
    };
    for cc in grid.chunk_coords() {
      let base = cc.0 * CHUNK_SIZE;
      let top = base + IVec3::splat(CHUNK_SIZE - 1);
      f.for_each_node(base, top, 4, &mut |origin, fill| match fill {
        NodeFill::Air => {}
        NodeFill::Solid(_) => {
          for dz in 0..4 {
            for dy in 0..4 {
              for dx in 0..4 {
                acc((origin + IVec3::new(dx, dy, dz)).as_vec3() + Vec3::splat(0.5));
              }
            }
          }
        }
        NodeFill::Mixed => {
          for dz in 0..4 {
            for dy in 0..4 {
              for dx in 0..4 {
                let p = origin + IVec3::new(dx, dy, dz);
                if grid.get_voxel(VoxelCoord::from_ivec3(p)).is_some() {
                  acc(p.as_vec3() + Vec3::splat(0.5));
                }
              }
            }
          }
        }
      });
    }
  }
  if n == 0 {
    return None;
  }
  let size = n as f32;
  let mass = size * density;
  let com = sum / size;
  let origin_moments = moments * density;
  let par = (Mat3::from_diagonal(Vec3::splat(com.length_squared()))
    - Mat3::from_cols(com * com.x, com * com.y, com * com.z))
    * mass;
  Some(MassProps { mass, com, inertia: origin_moments - par })
}

#[derive(Clone, Debug, Default)]
pub struct BodySet {
  pub pos: Vec<Vec3>,
  pub rot: Vec<Mat3>,
  pub lin_vel: Vec<Vec3>,
  pub ang_vel: Vec<Vec3>,
  pub pseudo_lin: Vec<Vec3>,
  pub pseudo_ang: Vec<Vec3>,
  pub inv_mass: Vec<f32>,
  pub inv_inertia_local: Vec<Mat3>,
  pub inv_inertia: Vec<Mat3>,
  pub scale: Vec<f32>,
  pub com: Vec<Vec3>,
  pub local_bounds: Vec<(IVec3, IVec3)>,
  pub grid_index: Vec<usize>,
  pub vox: Vec<ContactVoxels>,
  pub sleep_timer: Vec<f32>,
  pub sleeping: Vec<bool>,
  pub frozen: Vec<bool>,
}

impl BodySet {
  pub fn len(&self) -> usize {
    self.pos.len()
  }

  pub fn is_empty(&self) -> bool {
    self.pos.is_empty()
  }

  pub fn is_static(&self, i: usize) -> bool {
    self.inv_mass[i] == 0.0
  }

  pub fn push_static(&mut self, bounds: (IVec3, IVec3), grid_index: usize) -> usize {
    self.pos.push(Vec3::ZERO);
    self.rot.push(Mat3::IDENTITY);
    self.lin_vel.push(Vec3::ZERO);
    self.ang_vel.push(Vec3::ZERO);
    self.pseudo_lin.push(Vec3::ZERO);
    self.pseudo_ang.push(Vec3::ZERO);
    self.inv_mass.push(0.0);
    self.inv_inertia_local.push(Mat3::ZERO);
    self.inv_inertia.push(Mat3::ZERO);
    self.scale.push(1.0);
    self.com.push(Vec3::ZERO);
    self.local_bounds.push(bounds);
    self.grid_index.push(grid_index);
    self.vox.push(ContactVoxels::default());
    self.sleep_timer.push(0.0);
    self.sleeping.push(true);
    self.frozen.push(false);
    self.len() - 1
  }

  pub fn push_dynamic(
    &mut self,
    props: MassProps,
    bounds: (IVec3, IVec3),
    grid_index: usize,
    pos: Vec3,
    rot: Mat3,
    scale: f32,
  ) -> usize {
    let inv =
      if props.inertia.determinant().abs() > 1e-12 { props.inertia.inverse() } else { Mat3::ZERO };
    self.pos.push(pos);
    self.rot.push(rot);
    self.lin_vel.push(Vec3::ZERO);
    self.ang_vel.push(Vec3::ZERO);
    self.pseudo_lin.push(Vec3::ZERO);
    self.pseudo_ang.push(Vec3::ZERO);
    self.inv_mass.push(1.0 / props.mass);
    self.inv_inertia_local.push(inv);
    self.inv_inertia.push(Mat3::ZERO);
    self.scale.push(scale);
    self.com.push(props.com);
    self.local_bounds.push(bounds);
    self.grid_index.push(grid_index);
    self.vox.push(ContactVoxels::default());
    self.sleep_timer.push(0.0);
    self.sleeping.push(false);
    self.frozen.push(false);
    let i = self.len() - 1;
    self.update_inertia(i);
    i
  }

  pub fn remove(&mut self, i: usize) -> usize {
    debug_assert!(i < self.len());
    let last = self.len() - 1;
    self.pos.swap_remove(i);
    self.rot.swap_remove(i);
    self.lin_vel.swap_remove(i);
    self.ang_vel.swap_remove(i);
    self.pseudo_lin.swap_remove(i);
    self.pseudo_ang.swap_remove(i);
    self.inv_mass.swap_remove(i);
    self.inv_inertia_local.swap_remove(i);
    self.inv_inertia.swap_remove(i);
    self.scale.swap_remove(i);
    self.com.swap_remove(i);
    self.local_bounds.swap_remove(i);
    self.grid_index.swap_remove(i);
    self.vox.swap_remove(i);
    self.sleep_timer.swap_remove(i);
    self.sleeping.swap_remove(i);
    self.frozen.swap_remove(i);
    last
  }

  pub fn build_accel(&mut self, i: usize, grid: &VolumeGrid) {
    let field = Field::new(grid);
    self.vox[i] = ContactVoxels::build(&field, self.local_bounds[i]);
  }

  pub fn field_transform(&self, i: usize) -> VolumeTransform {
    VolumeTransform::new(
      self.pos[i] - self.rot[i] * (self.com[i] * self.scale[i]),
      self.rot[i],
      self.scale[i],
    )
  }

  pub fn set_pose_from_field(&mut self, i: usize, tr: VolumeTransform) {
    self.rot[i] = tr.rot;
    self.scale[i] = tr.scale;
    self.pos[i] = tr.pos + tr.rot * (self.com[i] * tr.scale);
    self.update_inertia(i);
  }

  pub fn update_inertia(&mut self, i: usize) {
    let r = self.rot[i];
    self.inv_inertia[i] = r * self.inv_inertia_local[i] * r.transpose();
  }

  pub fn update_all_inertia(&mut self) {
    for i in 0..self.len() {
      self.update_inertia(i);
    }
  }

  pub fn world_aabb(&self, i: usize) -> (Vec3, Vec3) {
    world_aabb_of(self.local_bounds[i], self.field_transform(i))
  }

  pub fn wake(&mut self, i: usize) {
    if self.is_static(i) || self.frozen[i] {
      return;
    }
    self.sleeping[i] = false;
    self.sleep_timer[i] = 0.0;
  }

  pub fn freeze(&mut self, i: usize) {
    if self.is_static(i) {
      return;
    }
    self.frozen[i] = true;
    self.sleeping[i] = true;
    self.sleep_timer[i] = 0.0;
    self.lin_vel[i] = Vec3::ZERO;
    self.ang_vel[i] = Vec3::ZERO;
    self.pseudo_lin[i] = Vec3::ZERO;
    self.pseudo_ang[i] = Vec3::ZERO;
  }

  pub fn is_frozen(&self, i: usize) -> bool {
    self.frozen[i]
  }

  pub fn put_to_sleep(&mut self, i: usize) {
    if self.is_static(i) {
      return;
    }
    self.sleeping[i] = true;
    self.lin_vel[i] = Vec3::ZERO;
    self.ang_vel[i] = Vec3::ZERO;
    self.pseudo_lin[i] = Vec3::ZERO;
    self.pseudo_ang[i] = Vec3::ZERO;
  }

  pub fn clear_pseudo(&mut self) {
    self.pseudo_lin.fill(Vec3::ZERO);
    self.pseudo_ang.fill(Vec3::ZERO);
  }

  pub fn apply_pseudo_impulse(&mut self, i: usize, p: Vec3, r: Vec3) {
    if self.sleeping[i] {
      return;
    }
    self.pseudo_lin[i] += self.inv_mass[i] * p;
    self.pseudo_ang[i] += self.inv_inertia[i] * r.cross(p);
  }

  pub fn integrate_pseudo(&mut self, i: usize, h: f32) -> bool {
    if self.sleeping[i] || (self.pseudo_lin[i] == Vec3::ZERO && self.pseudo_ang[i] == Vec3::ZERO) {
      return false;
    }
    let d = self.pseudo_lin[i] * h;
    self.pos[i] += d;
    let w = self.pseudo_ang[i] * h;
    let l = w.length();
    if l > 1e-12 {
      self.rot[i] = orthonormalize(Mat3::from_axis_angle(w / l, l) * self.rot[i]);
      self.update_inertia(i);
    }
    d.length_squared() > 0.0625
  }

  pub fn is_slow(&self, i: usize, lin: f32, ang: f32) -> bool {
    self.lin_vel[i].length_squared() <= lin * lin && self.ang_vel[i].length_squared() <= ang * ang
  }

  pub fn integrate_velocity(
    &mut self,
    i: usize,
    gravity: Vec3,
    h: f32,
    lin_damp: f32,
    ang_damp: f32,
  ) {
    if self.sleeping[i] {
      return;
    }
    self.lin_vel[i] += gravity * h;
    if lin_damp > 0.0 {
      self.lin_vel[i] /= 1.0 + h * lin_damp;
    }
    if ang_damp > 0.0 {
      self.ang_vel[i] /= 1.0 + h * ang_damp;
    }
  }

  pub fn integrate_position(&mut self, i: usize, h: f32) {
    if self.sleeping[i] {
      return;
    }
    self.pos[i] += self.lin_vel[i] * h;
    let w = self.ang_vel[i] * h;
    let l = w.length();
    if l > 1e-12 {
      self.rot[i] = orthonormalize(Mat3::from_axis_angle(w / l, l) * self.rot[i]);
    }
    self.update_inertia(i);
  }

  pub fn apply_impulse(&mut self, i: usize, p: Vec3, r: Vec3) {
    if self.sleeping[i] {
      return;
    }
    self.lin_vel[i] += self.inv_mass[i] * p;
    self.ang_vel[i] += self.inv_inertia[i] * r.cross(p);
  }

  pub fn point_velocity(&self, i: usize, r: Vec3) -> Vec3 {
    self.lin_vel[i] + self.ang_vel[i].cross(r)
  }

  pub fn pseudo_point_velocity(&self, i: usize, r: Vec3) -> Vec3 {
    self.pseudo_lin[i] + self.pseudo_ang[i].cross(r)
  }

  pub fn effective_mass(&self, i: usize, r: Vec3, dir: Vec3) -> f32 {
    let inv_i = self.inv_inertia[i];
    self.inv_mass[i] + dir.dot((inv_i * r.cross(dir)).cross(r))
  }
}

fn orthonormalize(m: Mat3) -> Mat3 {
  let x = m.x_axis.normalize_or_zero();
  let y = (m.y_axis - x * m.y_axis.dot(x)).normalize_or_zero();
  Mat3::from_cols(x, y, x.cross(y))
}
