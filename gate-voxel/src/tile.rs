//! Tile 与基元胞可变叶八叉树（P1.2）
//!
//! 存储模型（层级线性表，与 GPU 砖块图同构，P2.1 直接映射）：
//! - Tile：32³ 基元胞占用位掩码 + `HashMap<u16, Cell>`（仅存占用的基元胞）
//! - Cell：L0 整胞同色（uniform）或层级线性表 l1(8) → l2(64) → l3(512) → l4 brick(4096)
//! - Slot 三态：Empty / Leaf(palette) / Branch（下钻到下一级表）
//! - 线性寻址：父槽 s 的子槽 = `slots_of_parent`，无需指针
//!
//! 规范型不变式（canonical form，set/clear 后自动维护）：
//! 1. `uniform = Some` ⟺ 全部层级表为 None
//! 2. Branch 槽 ⟹ 其子区域确有数据；全空表即时删除
//! 3. 同色子区自动向上折叠为粗叶（「整块同色粗叶压缩存储」）

use std::collections::HashMap;

use crate::coords::{LEVEL_TABLE_AXIS, Level, MAX_LEVEL, VoxelPos};

/// 层级槽三态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Empty,
    /// 粗叶：该级整槽同色
    Leaf(u8),
    /// 分支：结构在下一级表
    Branch,
}

/// L4 砖块：4096 槽 palette 直存 + 占用位掩码
#[derive(Debug, Clone)]
pub struct Brick {
    pub occupancy: [u64; 64],
    pub palette: Box<[u8; 4096]>,
}

impl Brick {
    fn new() -> Self {
        Self {
            occupancy: [0; 64],
            palette: Box::new([0; 4096]),
        }
    }

    fn get(&self, slot: usize) -> Option<u8> {
        if self.occupancy[slot / 64] >> (slot % 64) & 1 == 1 {
            Some(self.palette[slot])
        } else {
            None
        }
    }

    fn set(&mut self, slot: usize, palette: u8) {
        self.occupancy[slot / 64] |= 1 << (slot % 64);
        self.palette[slot] = palette;
    }

    fn clear(&mut self, slot: usize) {
        self.occupancy[slot / 64] &= !(1 << (slot % 64));
    }

    fn is_empty(&self) -> bool {
        self.occupancy == [0; 64]
    }

    /// 堆上深尺寸（palette 槽表；occupancy 512B 为 Brick 内联部分，随 size_of 计）
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.palette)
    }
}

/// 基元胞：可变叶八叉树（层级线性表实现）
///
/// l4 用 `Option<Box<Brick>>`：Brick 520B 内联会把 uniform 粗叶胞撑到 ~560B
/// （Option<Brick> 无 niche 优化），装箱后空胞 ~40B——均匀粗叶场景（背景体素）
/// 内存 14×，P1.7 工作间预算断言依赖此布局
#[derive(Debug, Clone, Default)]
pub struct Cell {
    /// L0 整胞同色；Some 时全部层级表必须为 None
    pub uniform: Option<u8>,
    pub l1: Option<Box<[Slot; 8]>>,
    pub l2: Option<Box<[Slot; 64]>>,
    pub l3: Option<Box<[Slot; 512]>>,
    pub l4: Option<Box<Brick>>,
}

impl Cell {
    fn new() -> Self {
        Self::default()
    }

    /// 堆上深尺寸（层级表 + brick 含内联部分；自身 size_of 由容器层计）
    pub fn heap_bytes(&self) -> usize {
        let mut n = 0;
        if let Some(t) = &self.l1 {
            n += std::mem::size_of_val(&**t);
        }
        if let Some(t) = &self.l2 {
            n += std::mem::size_of_val(&**t);
        }
        if let Some(t) = &self.l3 {
            n += std::mem::size_of_val(&**t);
        }
        if let Some(b) = &self.l4 {
            n += std::mem::size_of::<Brick>() + b.heap_bytes();
        }
        n
    }

    /// 槽状态查询；表不存在 = 整级无数据（Empty）
    fn slot_state(&self, level: Level, slot: usize) -> Slot {
        match level {
            1 => self.l1.as_ref().map(|t| t[slot]).unwrap_or(Slot::Empty),
            2 => self.l2.as_ref().map(|t| t[slot]).unwrap_or(Slot::Empty),
            3 => self.l3.as_ref().map(|t| t[slot]).unwrap_or(Slot::Empty),
            4 => match self.l4.as_ref().and_then(|b| b.get(slot)) {
                Some(p) => Slot::Leaf(p),
                None => Slot::Empty,
            },
            _ => unreachable!("level 0 has no slot table"),
        }
    }

    /// 确保到 level 的表链存在，沿途把上级对应槽置 Branch。
    /// 调用前 cell 必须非 uniform（调用方处理）
    fn ensure_tables(&mut self, level: Level, slots: &[usize; 3]) {
        if self.l1.is_none() {
            self.l1 = Some(Box::new([Slot::Empty; 8]));
        }
        for l in 1..level as usize {
            let parent_slot = slots[l - 1];
            match l {
                1 => {
                    self.l1.as_mut().unwrap()[parent_slot] = Slot::Branch;
                    if self.l2.is_none() {
                        self.l2 = Some(Box::new([Slot::Empty; 64]));
                    }
                }
                2 => {
                    self.l2.as_mut().unwrap()[parent_slot] = Slot::Branch;
                    if self.l3.is_none() {
                        self.l3 = Some(Box::new([Slot::Empty; 512]));
                    }
                }
                3 => {
                    self.l3.as_mut().unwrap()[parent_slot] = Slot::Branch;
                    if self.l4.is_none() {
                        self.l4 = Some(Box::new(Brick::new()));
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    /// 清除 level 槽区域的全部更深数据（不修改该槽自身）
    fn clear_deeper(&mut self, level: Level, slot: usize) {
        if level >= MAX_LEVEL {
            return;
        }
        let children = slots_of_parent(level, slot);
        match level {
            1 => {
                if self.l2.is_none() {
                    return;
                }
                // 深层受影响区 = 各 l2 子槽对应的 l3 槽
                let l3_slots: Vec<usize> = children
                    .iter()
                    .flat_map(|&s2| slots_of_parent(2, s2))
                    .collect();
                if let Some(t2) = self.l2.as_mut() {
                    for &c in &children {
                        t2[c] = Slot::Empty;
                    }
                }
                self.clear_deeper_l2(&l3_slots);
                if self
                    .l2
                    .as_ref()
                    .is_some_and(|t| t.iter().all(|s| *s == Slot::Empty))
                {
                    self.l2 = None;
                }
            }
            2 => {
                if self.l3.is_none() {
                    return;
                }
                if let Some(t3) = self.l3.as_mut() {
                    for &c in &children {
                        t3[c] = Slot::Empty;
                    }
                }
                self.clear_deeper_l3(&children);
                if self
                    .l3
                    .as_ref()
                    .is_some_and(|t| t.iter().all(|s| *s == Slot::Empty))
                {
                    self.l3 = None;
                }
            }
            3 => {
                let Some(brick) = self.l4.as_mut() else {
                    return;
                };
                for &c in &children {
                    brick.clear(c);
                }
                if brick.is_empty() {
                    self.l4 = None;
                }
            }
            _ => unreachable!(),
        }
    }

    /// 清 l3 槽状态 + 其 l4 深层，表空即删
    fn clear_deeper_l2(&mut self, l3_slots: &[usize]) {
        if self.l3.is_none() {
            return;
        }
        if let Some(t3) = self.l3.as_mut() {
            for &c in l3_slots {
                t3[c] = Slot::Empty;
            }
        }
        self.clear_deeper_l3(l3_slots);
        if self
            .l3
            .as_ref()
            .is_some_and(|t| t.iter().all(|s| *s == Slot::Empty))
        {
            self.l3 = None;
        }
    }

    /// 清 l3 槽对应的 l4 占用位，brick 空即删
    fn clear_deeper_l3(&mut self, l3_slots: &[usize]) {
        let Some(brick) = self.l4.as_mut() else {
            return;
        };
        for &s3 in l3_slots {
            for &c4 in &slots_of_parent(3, s3) {
                brick.clear(c4);
            }
        }
        if brick.is_empty() {
            self.l4 = None;
        }
    }

    /// 从 (level, slot) 向上折叠同色子区（规范型维护）；只折叠到 L1
    fn coalesce_up(&mut self, level: Level, slot: usize) {
        let mut level = level;
        let mut slot = slot;
        while level > 1 {
            let Some((p_level, p_slot)) = parent_slot_of(level, slot) else {
                break;
            };
            if !self.try_fold(p_level, p_slot) {
                return;
            }
            level = p_level;
            slot = p_slot;
        }
        // L1 表全同色 → 整胞折叠为 uniform
        if let Some(t1) = self.l1.as_ref()
            && let Some(Slot::Leaf(p)) = t1.iter().next()
            && t1.iter().all(|s| matches!(s, Slot::Leaf(q) if *q == *p))
        {
            *self = Cell {
                uniform: Some(*p),
                ..Default::default()
            };
        }
    }

    /// 尝试把 p_level 槽 p_slot 的 8 个子槽折叠为该槽 Leaf；返回是否发生折叠
    fn try_fold(&mut self, p_level: Level, p_slot: usize) -> bool {
        debug_assert!((1..=3).contains(&p_level));
        let children = slots_of_parent(p_level, p_slot);
        // 判定阶段（不可变）
        let first = match self.slot_state(p_level + 1, children[0]) {
            Slot::Leaf(p) => p,
            _ => return false,
        };
        if !children
            .iter()
            .all(|&c| self.slot_state(p_level + 1, c) == Slot::Leaf(first))
        {
            return false;
        }
        // 写入阶段（可变）
        match p_level {
            1 => self.l1.as_mut().unwrap()[p_slot] = Slot::Leaf(first),
            2 => self.l2.as_mut().unwrap()[p_slot] = Slot::Leaf(first),
            3 => self.l3.as_mut().unwrap()[p_slot] = Slot::Leaf(first),
            _ => unreachable!(),
        }
        // 深层区域数据归属上移，清除残留
        self.clear_deeper(p_level, p_slot);
        true
    }
}

/// 父槽 s 在下一级表中的 8 个子槽线性索引
fn slots_of_parent(level: Level, slot: usize) -> [usize; 8] {
    let axis = LEVEL_TABLE_AXIS[level as usize];
    let (x, y, z) = (slot % axis, slot / axis % axis, slot / (axis * axis));
    let ca = axis * 2;
    let mut out = [0usize; 8];
    for (k, o) in out.iter_mut().enumerate() {
        let (i, j, kk) = (k & 1, (k >> 1) & 1, k >> 2);
        *o = (x * 2 + i) + (y * 2 + j) * ca + (z * 2 + kk) * ca * ca;
    }
    out
}

/// 反查：(level, slot) 的父级槽
fn parent_slot_of(level: Level, slot: usize) -> Option<(Level, usize)> {
    if level == 0 {
        return None;
    }
    let axis = LEVEL_TABLE_AXIS[level as usize];
    let (x, y, z) = (slot % axis, slot / axis % axis, slot / (axis * axis));
    let pa = axis / 2;
    Some((level - 1, x / 2 + (y / 2) * pa + (z / 2) * pa * pa))
}

/// pos 沿途各级表链的槽索引：[0]=L1 槽，[1]=L2 槽，[2]=L3 槽
fn ancestor_slots(pos: &VoxelPos) -> [usize; 3] {
    let fine = pos.fine_min();
    let mut slots = [0usize; 3];
    for l in 1..pos.level as usize {
        slots[l - 1] = VoxelPos::from_fine(fine, l as Level).slot_index();
    }
    slots
}

/// 打散 uniform 后的回填：各级已建表中 Empty 槽 = Leaf(old)。
/// Branch 槽（ensure_tables 已置）保留；L4 brick 只占目标位（Leaf@l3 已承载语义）
fn backfill_tables(cell: &mut Cell, level: Level, old: u8) {
    let fill = |t: &mut [Slot]| {
        for s in t.iter_mut() {
            if *s == Slot::Empty {
                *s = Slot::Leaf(old);
            }
        }
    };
    if let Some(t) = cell.l1.as_mut() {
        fill(&mut t[..]);
    }
    if level >= 2
        && let Some(t) = cell.l2.as_mut()
    {
        fill(&mut t[..]);
    }
    if level >= 3
        && let Some(t) = cell.l3.as_mut()
    {
        fill(&mut t[..]);
    }
}

/// Tile：32³ 基元胞 + 元件层占位
#[derive(Debug, Clone)]
pub struct Tile {
    /// 基元胞占用位掩码，bit = cell_index（x + y*32 + z*1024）
    pub occupancy: [u64; 512],
    /// 仅存占用的基元胞
    pub cells: HashMap<u16, Cell>,
    /// 元件层占位（P5 洪泛后启用；粒度与布局 P5 定稿，先不分配）
    pub comp_layer: Option<Box<[u16]>>,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            occupancy: [0; 512],
            cells: HashMap::new(),
            comp_layer: None,
        }
    }
}

impl Tile {
    /// 堆上深尺寸（cells 哈希表 + 各基元胞层级表；occupancy 4KB 内联部分由 size_of::<Tile> 计）
    ///
    /// hashbrown 桶开销 ≈ capacity × (entry + 1 控制字节) × 8/7（capacity=7/8 负载因子的倒数），
    /// 估值只高不低，用于内存预算断言是保守方向
    pub fn heap_bytes(&self) -> usize {
        let map = self.cells.capacity() * (std::mem::size_of::<(u16, Cell)>() + 1) / 7 * 8;
        let cells: usize = self.cells.values().map(Cell::heap_bytes).sum();
        let comp = self
            .comp_layer
            .as_ref()
            .map(|l| std::mem::size_of_val(&**l))
            .unwrap_or(0);
        map + cells + comp
    }

    /// 占用基元胞数（统计用）
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    pub fn cell(&self, idx: u16) -> Option<&Cell> {
        if self.occupancy[idx as usize / 64] >> (idx as usize % 64) & 1 == 1 {
            self.cells.get(&idx)
        } else {
            None
        }
    }

    pub fn cell_mut(&mut self, idx: u16) -> &mut Cell {
        self.occupancy[idx as usize / 64] |= 1 << (idx as usize % 64);
        self.cells.entry(idx).or_default()
    }

    /// 基元胞是否完全无数据（无任何叶/表/uniform）
    pub fn is_cell_empty(&self, idx: u16) -> bool {
        match self.cells.get(&idx) {
            None => true,
            Some(c) => {
                c.uniform.is_none()
                    && c.l1.is_none()
                    && c.l2.is_none()
                    && c.l3.is_none()
                    && c.l4.is_none()
            }
        }
    }

    pub fn remove_cell_if_empty(&mut self, idx: u16) {
        if self.is_cell_empty(idx) {
            self.occupancy[idx as usize / 64] &= !(1 << (idx as usize % 64));
            self.cells.remove(&idx);
        }
    }

    /// 包含该点的最深叶颜色（Branch 一路下钻，遇粗叶即返）
    pub fn get_leaf(&self, pos: &VoxelPos) -> Option<u8> {
        let cell = self.cell(pos.cell_index())?;
        if let Some(p) = cell.uniform {
            return Some(p);
        }
        for level in 1..=MAX_LEVEL {
            let slot = VoxelPos::from_fine(pos.fine_min(), level).slot_index();
            match cell.slot_state(level, slot) {
                Slot::Leaf(p) => return Some(p),
                Slot::Empty => return None,
                Slot::Branch => continue,
            }
        }
        None
    }

    /// 区域全同色则返回颜色（含下钻聚合；区域无体素返回 None）
    pub fn get_uniform(&self, pos: &VoxelPos) -> Option<u8> {
        let cell = self.cell(pos.cell_index())?;
        if let Some(p) = cell.uniform {
            return Some(p);
        }
        if pos.level == 0 {
            return None;
        }
        self.uniform_at(cell, pos.level, pos.slot_index())
    }

    fn uniform_at(&self, cell: &Cell, level: Level, slot: usize) -> Option<u8> {
        match cell.slot_state(level, slot) {
            Slot::Leaf(p) => Some(p),
            Slot::Empty => None,
            Slot::Branch => {
                if level == MAX_LEVEL {
                    return None;
                }
                let children = slots_of_parent(level, slot);
                let mut result = None;
                for &c in &children {
                    let cur = self.uniform_at(cell, level + 1, c)?;
                    match result {
                        None => result = Some(cur),
                        Some(p) if p == cur => {}
                        _ => return None,
                    }
                }
                result
            }
        }
    }

    /// 写叶（破坏性覆盖更深结构）+ 同色折叠，返回是否实际修改
    pub fn set_leaf(&mut self, pos: &VoxelPos, palette: u8) -> bool {
        debug_assert!(palette != 0, "palette 0 is AIR; use clear_voxel");
        let idx = pos.cell_index();
        let cell = self.cell_mut(idx);

        if pos.level == 0 {
            if cell.uniform == Some(palette) {
                return false;
            }
            *cell = Cell {
                uniform: Some(palette),
                ..Default::default()
            };
            return true;
        }

        let slot = pos.slot_index();
        let slots = ancestor_slots(pos);
        // uniform 打散：建链 + 回填旧色（保持其余区域语义）
        let old = cell.uniform.take();
        cell.ensure_tables(pos.level, &slots);
        if let Some(old) = old {
            backfill_tables(cell, pos.level, old);
            if pos.level == MAX_LEVEL {
                cell.l4.as_mut().unwrap().set(slot, old);
            }
        }
        // 破坏性覆盖：先清更深区域，再写本槽
        cell.clear_deeper(pos.level, slot);

        let changed = match pos.level {
            1 => {
                let t = cell.l1.as_mut().unwrap();
                if t[slot] == Slot::Leaf(palette) {
                    false
                } else {
                    t[slot] = Slot::Leaf(palette);
                    true
                }
            }
            2 => {
                let t = cell.l2.as_mut().unwrap();
                if t[slot] == Slot::Leaf(palette) {
                    false
                } else {
                    t[slot] = Slot::Leaf(palette);
                    true
                }
            }
            3 => {
                let t = cell.l3.as_mut().unwrap();
                if t[slot] == Slot::Leaf(palette) {
                    false
                } else {
                    t[slot] = Slot::Leaf(palette);
                    true
                }
            }
            4 => {
                let brick = cell.l4.as_mut().unwrap();
                if brick.get(slot) == Some(palette) {
                    false
                } else {
                    brick.set(slot, palette);
                    true
                }
            }
            _ => unreachable!(),
        };
        if changed {
            cell.coalesce_up(pos.level, slot);
        }
        changed
    }

    /// 清除体素（该槽置空；父级 Branch 随子区变空而退化）
    pub fn clear_voxel(&mut self, pos: &VoxelPos) -> bool {
        let idx = pos.cell_index();
        if self.is_cell_empty(idx) {
            return false;
        }
        if pos.level == 0 {
            let cell = self.cells.get_mut(&idx).unwrap();
            *cell = Cell::new();
            self.remove_cell_if_empty(idx);
            return true;
        }

        let slot = pos.slot_index();
        // uniform 态：目标区域必然有数据，先打散再清
        let has_data = {
            let cell = self.cells.get_mut(&idx).unwrap();
            if cell.uniform.is_some() {
                let slots = ancestor_slots(pos);
                let old = cell.uniform.take().unwrap();
                cell.ensure_tables(pos.level, &slots);
                backfill_tables(cell, pos.level, old);
                if pos.level == MAX_LEVEL {
                    cell.l4.as_mut().unwrap().set(slot, old);
                }
                true
            } else {
                cell.slot_state(pos.level, slot) != Slot::Empty
            }
        };
        if !has_data {
            return false;
        }

        {
            let cell = self.cells.get_mut(&idx).unwrap();
            cell.clear_deeper(pos.level, slot);
            match pos.level {
                1 => cell.l1.as_mut().unwrap()[slot] = Slot::Empty,
                2 => cell.l2.as_mut().unwrap()[slot] = Slot::Empty,
                3 => cell.l3.as_mut().unwrap()[slot] = Slot::Empty,
                4 => {
                    let brick = cell.l4.as_mut().unwrap();
                    brick.clear(slot);
                    if brick.is_empty() {
                        cell.l4 = None;
                    }
                }
                _ => unreachable!(),
            }
            Self::shrink_up(cell, pos.level, slot);
            // 兜底压缩：自深向浅清除空 brick 与空表（不变式 2）
            if cell.l4.as_ref().is_some_and(|b| b.is_empty()) {
                cell.l4 = None;
            }
            if cell
                .l3
                .as_ref()
                .is_some_and(|t| t.iter().all(|s| *s == Slot::Empty))
            {
                cell.l3 = None;
            }
            if cell
                .l2
                .as_ref()
                .is_some_and(|t| t.iter().all(|s| *s == Slot::Empty))
            {
                cell.l2 = None;
            }
        }
        self.remove_cell_if_empty(idx);
        true
    }

    /// 父级 Branch 槽在子区全空后退化为 Empty；全空表删除（逐级向上）
    fn shrink_up(cell: &mut Cell, level: Level, slot: usize) {
        let mut level = level;
        let mut slot = slot;
        while level > 1 {
            let Some((p_level, p_slot)) = parent_slot_of(level, slot) else {
                return;
            };
            let has_deeper = slots_of_parent(p_level, p_slot)
                .iter()
                .any(|&c| cell.slot_state(p_level + 1, c) != Slot::Empty);
            if has_deeper {
                return;
            }
            match p_level {
                1 => cell.l1.as_mut().unwrap()[p_slot] = Slot::Empty,
                2 => cell.l2.as_mut().unwrap()[p_slot] = Slot::Empty,
                3 => cell.l3.as_mut().unwrap()[p_slot] = Slot::Empty,
                _ => unreachable!(),
            }
            // 全空表删除
            match p_level {
                1 => {
                    if cell
                        .l1
                        .as_ref()
                        .is_some_and(|t| t.iter().all(|s| *s == Slot::Empty))
                    {
                        cell.l1 = None;
                    }
                }
                2 => {
                    if cell
                        .l2
                        .as_ref()
                        .is_some_and(|t| t.iter().all(|s| *s == Slot::Empty))
                    {
                        cell.l2 = None;
                    }
                }
                3 => {
                    if cell
                        .l3
                        .as_ref()
                        .is_some_and(|t| t.iter().all(|s| *s == Slot::Empty))
                    {
                        cell.l3 = None;
                    }
                }
                _ => unreachable!(),
            }
            level = p_level;
            slot = p_slot;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    #[test]
    fn slots_of_parent_linear_layout() {
        // L1(axis=2) 槽 0 的子槽 = L2(axis=4) 的 0,1,4,5,16,17,20,21
        assert_eq!(slots_of_parent(1, 0), [0, 1, 4, 5, 16, 17, 20, 21]);
        // L1 槽 3（x=1,y=1,z=0）的子槽含 x∈{2,3}, y∈{2,3}, z∈{0,1}
        let children = slots_of_parent(1, 3);
        assert_eq!(children[0], 2 + 2 * 4); // x=2,y=2,z=0
    }

    #[test]
    fn uniform_cell_roundtrip() {
        let mut tile = Tile::default();
        let pos = VoxelPos::from_fine(IVec3::new(0, 0, 0), 0);
        assert!(tile.set_leaf(&pos, 5));
        assert_eq!(tile.get_leaf(&pos), Some(5));
        // 整胞 uniform：cell 0 内任意点都是 5（fine 0..16）
        let inner = VoxelPos::from_fine(IVec3::new(15, 15, 15), 4);
        assert_eq!(tile.get_leaf(&inner), Some(5));
        assert_eq!(tile.get_uniform(&inner.with_level(2)), Some(5));
        // 折叠验证：cell 回到 uniform 形态
        let cell = tile.cell(0).unwrap();
        assert_eq!(cell.uniform, Some(5));
        assert!(cell.l1.is_none() && cell.l2.is_none() && cell.l3.is_none() && cell.l4.is_none());
    }

    #[test]
    fn set_refine_over_uniform() {
        let mut tile = Tile::default();
        tile.set_leaf(&VoxelPos::from_fine(IVec3::ZERO, 0), 1);
        // 细分其中一个 L3（0.5cm）子区（fine 2..4 立方，cell 0 内）
        let pos3 = VoxelPos::from_fine(IVec3::new(2, 2, 2), 3);
        assert!(tile.set_leaf(&pos3, 2));
        assert_eq!(tile.get_leaf(&pos3), Some(2));
        // 其余区域仍是粗叶 1（同 cell 不同槽）
        assert_eq!(
            tile.get_leaf(&VoxelPos::from_fine(IVec3::new(4, 0, 0), 3)),
            Some(1)
        );
        let cell = tile.cell(0).unwrap();
        assert!(cell.uniform.is_none());
        assert!(cell.l3.is_some());
    }

    #[test]
    fn l4_brick_write_and_coalesce() {
        let mut tile = Tile::default();
        // 写满一个 L3 槽的 8 个 L4 位（同色）→ 应折叠成 L3 粗叶
        let base = IVec3::new(0, 0, 0);
        for i in 0..8i32 {
            let off = IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            let pos = VoxelPos::from_fine(base + off, 4);
            assert!(tile.set_leaf(&pos, 7));
        }
        let pos3 = VoxelPos::from_fine(base, 3);
        let cell = tile.cell(0).unwrap();
        // l3 槽应已折叠为 Leaf(7)，brick 占用位已清
        let l3_slot = pos3.slot_index();
        assert_eq!(cell.l3.as_ref().unwrap()[l3_slot], Slot::Leaf(7));
        assert!(cell.l4.as_ref().map(|b| b.is_empty()).unwrap_or(true));
        assert_eq!(tile.get_leaf(&pos3), Some(7));
    }

    #[test]
    fn coarse_write_clobbers_deeper() {
        let mut tile = Tile::default();
        // L4 写两个不同色
        tile.set_leaf(&VoxelPos::from_fine(IVec3::new(0, 0, 0), 4), 1);
        tile.set_leaf(&VoxelPos::from_fine(IVec3::new(1, 0, 0), 4), 2);
        assert_eq!(
            tile.get_leaf(&VoxelPos::from_fine(IVec3::new(1, 0, 0), 4)),
            Some(2)
        );
        // L3 粗写覆盖（破坏性）
        let pos3 = VoxelPos::from_fine(IVec3::new(0, 0, 0), 3);
        tile.set_leaf(&pos3, 3);
        assert_eq!(
            tile.get_leaf(&VoxelPos::from_fine(IVec3::new(1, 0, 0), 4)),
            Some(3)
        );
        // 粗写区域内、此前无 L4 点的位置也由粗叶承载
        assert_eq!(
            tile.get_leaf(&VoxelPos::from_fine(IVec3::new(1, 1, 1), 4)),
            Some(3)
        );
        // 粗写区域外仍是空
        assert_eq!(
            tile.get_leaf(&VoxelPos::from_fine(IVec3::new(7, 7, 7), 4)),
            None
        );
    }

    #[test]
    fn clear_and_shrink() {
        let mut tile = Tile::default();
        tile.set_leaf(&VoxelPos::from_fine(IVec3::ZERO, 0), 9);
        // 清一个 L4 点 → uniform 被打散成表结构
        let pos4 = VoxelPos::from_fine(IVec3::new(15, 15, 15), 4);
        assert!(tile.clear_voxel(&pos4));
        assert_eq!(tile.get_leaf(&pos4), None);
        assert_eq!(
            tile.get_leaf(&VoxelPos::from_fine(IVec3::new(0, 0, 0), 4)),
            Some(9)
        );
        // 清掉整体（L0）
        assert!(tile.clear_voxel(&VoxelPos::from_fine(IVec3::ZERO, 0)));
        assert!(tile.is_cell_empty(0));
        assert!(tile.cell(0).is_none());
    }

    #[test]
    fn clear_l4_shrinks_branch_chain() {
        let mut tile = Tile::default();
        // 单个 L4 点 → 表链 l1..l4 全建
        let pos4 = VoxelPos::from_fine(IVec3::new(15, 15, 15), 4);
        tile.set_leaf(&pos4, 3);
        assert!(tile.cell(0).unwrap().l4.is_some());
        // 清掉 → 表链应逐级收缩消失，cell 回到空
        assert!(tile.clear_voxel(&pos4));
        assert!(tile.is_cell_empty(0));
        tile.remove_cell_if_empty(0);
        assert!(tile.cell(0).is_none());
        assert_eq!(tile.occupancy, [0; 512]);
    }

    #[test]
    fn set_same_color_is_noop() {
        let mut tile = Tile::default();
        let pos = VoxelPos::from_fine(IVec3::new(4, 4, 4), 2);
        assert!(tile.set_leaf(&pos, 5));
        assert!(!tile.set_leaf(&pos, 5));
    }
}
