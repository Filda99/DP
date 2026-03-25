"""
Environment System

Creates environmental features like obstacles, terrain, weather effects.
Manages cities, forests, wind, and other environmental factors.
Includes wildfire simulation capabilities using precise rasterization.
"""

import pybullet as p
import numpy as np
import random
import sys
import os
import time
from shapely.geometry import Point

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.fire_grid import FireGrid
from src.grid_mapper import GridMapper

# Import matplotlib path for fast rasterization
from matplotlib.path import Path as MplPath


class Environment:
    """Environment system with obstacles, terrain, and weather."""
    
    def __init__(self):
        """Initialize environment."""
        self.obstacles = []
        self.terrain_zones = []
        self.fire_visual_objects = []
        
        # Weather & Wind
        self._initialize_random_wind()
        self.weather = {
            'wind_velocity': self.wind_velocity,
            'visibility': 1000.0,
            'precipitation': 0.0
        }
        
        self.wind_change_timer = 0.0
        self.wind_change_interval = random.uniform(5.0, 15.0)
        self.target_wind = self.wind_velocity.copy()
        
        # We need to define the size of the world (e.g., 200m x 200m)
        self.grid_width = 200.0
        self.grid_height = 200.0
        self.cell_size = 2.0 # 2 meters per cell
        
        # 1. Create Mapper
        self.grid_mapper = GridMapper(
            grid_width_m=self.grid_width, 
            grid_height_m=self.grid_height, 
            cell_size_m=self.cell_size
        )
        
        # 2. Create Fire Grid
        self.fire_grid = FireGrid(
            H=self.grid_mapper.grid_height_cells,
            W=self.grid_mapper.grid_width_cells,
            dt=0.1 # Simulation time step for fire
        )

        self.fire_time_accumulator = 0.0
        self.fire_enabled = False
        
        # Storage for terrain data if loaded before fire grid is initialized
        self._pending_terrain_data = None

        self.refill_zone = None # {'position': [x,y,z], 'radius': r}

    # ============================================================================
    # RASTERIZATION (FUEL MAP GENERATION)
    # ============================================================================

    def rasterize_terrain_layers(self, gdf_water, gdf_buildings, gdf_forest):
        """
        Converts vector data (GeoDataFrames) into the FireGrid fuel map.
        Uses Painter's Algorithm: Grass (Default) -> Forest -> Buildings -> Water.
        OPTIMIZED: Uses matplotlib.path for fast C-based rasterization.
        """
        if self.fire_grid is None or self.grid_mapper is None:
            print("📝 FireGrid not initialized yet. Storing terrain data for later rasterization.")
            self._pending_terrain_data = (gdf_water, gdf_buildings, gdf_forest)
            return

        # print("🔥 Rasterizing terrain into FireGrid (Optimized Painter's Algorithm)...")
        t_start = time.time()
        
        H, W = self.fire_grid.H, self.fire_grid.W
        
        # Fuel properties: (fuel_level, burn_rate_per_sq_meter)
        # Burn rate per square meter - stejný burn rate na m2 bez ohledu na cell size
        
        # Reálné burn times pro 1x1m buňku:
        # Tráva: 30 sekund -> burn_rate = fuel_level / 30s = 0.3 / 30 = 0.01 per second per m2
        # Les: 2 minuty -> burn_rate = 0.8 / 120 = 0.0067 per second per m2  
        # Budova: 10 minut -> burn_rate = 0.9 / 600 = 0.0015 per second per m2
        
        BURN_RATE_GRASS_PER_M2 = 0.01      # 30s pro 1x1m buňku
        BURN_RATE_FOREST_PER_M2 = 0.0067    # 2 min pro 1x1m buňku 
        BURN_RATE_BUILDING_PER_M2 = 0.0015  # 10 min pro 1x1m buňku
        
        # Scale burn rate podle cell size - větší buňka horí proporcionálně déle
        cell_area = cell_size * cell_size  # m2
        
        FUEL_WATER = (0.0, 0.0) 
        FUEL_BUILDING = (0.9, BURN_RATE_BUILDING_PER_M2 / cell_area)  # Scale podle plochy
        FUEL_FOREST = (0.8, BURN_RATE_FOREST_PER_M2 / cell_area)     # Scale podle plochy
        FUEL_GRASS = (0.3, BURN_RATE_GRASS_PER_M2 / cell_area)       # Scale podle plochy        
        # Debug info - ukažme burn times
        grass_burn_time = FUEL_GRASS[0] / FUEL_GRASS[1] if FUEL_GRASS[1] > 0 else 0
        forest_burn_time = FUEL_FOREST[0] / FUEL_FOREST[1] if FUEL_FOREST[1] > 0 else 0 
        # 1. BASE LAYER: GRASS/OPEN (Default)
        self.fire_grid.F[:] = FUEL_GRASS[0]
        self.fire_grid.fuel_burn_rate[:] = FUEL_GRASS[1]
        
        # Pre-calculate mapping constants to speed up loop
        origin_x = self.grid_mapper.origin_x
        origin_y = self.grid_mapper.origin_y
        cell_size = self.grid_mapper.cell_size_m
        
        def burn_polygons_to_grid(gdf, fuel_val, rate_val, label):
            """Helper to burn polygons into the grid using vectorized checks."""
            if gdf is None or len(gdf) == 0:
                return 0
            
            count = 0
            
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                
                # Handle MultiPolygons by breaking them down
                if geom.geom_type == 'MultiPolygon':
                    polys = list(geom.geoms)
                elif geom.geom_type == 'Polygon':
                    polys = [geom]
                else:
                    continue
                    
                for poly in polys:
                    # Get bounds
                    minx, miny, maxx, maxy = poly.bounds
                    
                    # Convert bounds to grid indices
                    # Use ceil for max to ensure coverage
                    j_min = int((minx - origin_x) / cell_size)
                    i_min = int((miny - origin_y) / cell_size)
                    j_max = int(np.ceil((maxx - origin_x) / cell_size))
                    i_max = int(np.ceil((maxy - origin_y) / cell_size))
                    
                    # Clip to grid dimensions
                    j_min = max(0, min(j_min, W))
                    j_max = max(0, min(j_max, W))
                    i_min = max(0, min(i_min, H))
                    i_max = max(0, min(i_max, H))
                    
                    # Skip if outside grid
                    if j_min >= j_max or i_min >= i_max:
                        continue
                        
                    # 1. Create coordinate grid for the Bounding Box
                    x_coords = origin_x + (np.arange(j_min, j_max) + 0.5) * cell_size
                    y_coords = origin_y + (np.arange(i_min, i_max) + 0.5) * cell_size
                    
                    xv, yv = np.meshgrid(x_coords, y_coords) 
                    
                    # Flatten to list of points (N, 2)
                    points = np.vstack((xv.flatten(), yv.flatten())).T
                    
                    # 2. Vectorized Point-in-Polygon Check
                    path = MplPath(list(poly.exterior.coords))
                    mask_flat = path.contains_points(points)
                    mask = mask_flat.reshape(xv.shape)
                    
                    # 3. Apply mask to FireGrid
                    fuel_slice = self.fire_grid.F[i_min:i_max, j_min:j_max]
                    rate_slice = self.fire_grid.fuel_burn_rate[i_min:i_max, j_min:j_max]
                    
                    fuel_slice[mask] = fuel_val
                    rate_slice[mask] = rate_val
                    
                    count += np.sum(mask)
                    
            return count

        # 2. APPLY LAYERS IN PRIORITY ORDER
        cells_forest = burn_polygons_to_grid(gdf_forest, *FUEL_FOREST, "Forest")
        # print(f"   🌲 Applied Forest layer ({cells_forest} cells)")
        
        cells_bld = burn_polygons_to_grid(gdf_buildings, *FUEL_BUILDING, "Buildings")
        # print(f"   🏢 Applied Building layer ({cells_bld} cells)")
        
        cells_water = burn_polygons_to_grid(gdf_water, *FUEL_WATER, "Water")
        # print(f"   💧 Applied Water layer ({cells_water} cells)")
        
        # print(f"✅ Terrain rasterization complete in {time.time() - t_start:.2f}s")

    # ============================================================================
    # PYBULLET OBJECT CREATION (VISUALS)
    # ============================================================================
    
    def create_ground(self):
        ground_id = p.loadURDF("../urdf/plane.urdf")
        return ground_id
    
    def add_city_block(self, position, size=[5, 5, 10], color=[0.7, 0.7, 0.7, 1]):
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        building_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[position[0], position[1], size[2]/2])
        self.obstacles.append({'id': building_id, 'type': 'city_block', 'position': position, 'size': size, 'bounds': self._calculate_bounds(position, size)})
        return building_id
    
    def add_forest_area(self, center, radius, tree_count=20):
        forest = {'type': 'forest', 'center': center, 'radius': radius, 'bounds': {'min': [center[0]-radius, center[1]-radius, 0], 'max': [center[0]+radius, center[1]+radius, 20]}}
        self.terrain_zones.append(forest)
    
    def add_lake(self, center, radius):
        lake_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=0.1, rgbaColor=[0.1, 0.5, 0.9, 0.8])
        p.createMultiBody(baseVisualShapeIndex=lake_visual, basePosition=[center[0], center[1], 0.05])
        self.terrain_zones.append({'type': 'lake', 'center': center, 'radius': radius})

    # ============================================================================
    # FIRE SIMULATION
    # ============================================================================
    
    def ignite_fire(self, x: float, y: float, intensity: float = 1.0):
        """
        Ignites a fire at the specified world coordinates (x, y).
        
        Args:
            x (float): X position in world meters
            y (float): Y position in world meters
            intensity (float): Initial fire intensity [0.0, 1.0]
        """
        # 1. Check if Mapper and Grid are initialized
        if not hasattr(self, 'grid_mapper') or not hasattr(self, 'fire_grid'):
            print("⚠️ Fire Grid components not ready. Cannot ignite.")
            return

        # 2. Convert World Coordinates -> Grid Indices
        # The wrapper gives us (x, y), we need (row, col)
        r, c = self.grid_mapper.world_to_cell((x, y))
        
        # 3. Update the FireGrid state directly
        # We access the arrays B (Burning) and I (Intensity)
        if 0 <= r < self.fire_grid.H and 0 <= c < self.fire_grid.W:
            self.fire_grid.B[r, c] = True        # Set Burning flag to True
            self.fire_grid.I[r, c] = intensity   # Set Intensity
            # print(f"🔥 Ignition confirmed at world=({x:.1f}, {y:.1f}) -> grid=[{r}, {c}]")
        else:
            print(f"⚠️ Ignition failed: Coordinates ({x:.1f}, {y:.1f}) are out of simulation bounds.")

    def enable_fire_simulation(self, grid_width_m=100, grid_height_m=100, cell_size_m=2.0, 
                             dt=0.1, alpha=1.0, k_wind=1.5, wind_dir=0.0, lazy_fuel=False):
        """
        Enable wildfire simulation.
        """
        # print(f"🔥 Enabling Fire Simulation...")
        self.grid_mapper = GridMapper(grid_width_m, grid_height_m, cell_size_m)
        H, W = self.grid_mapper.get_grid_dimensions()
        
        wind_velocity = self.weather['wind_velocity'][:2]
        wind_speed = np.linalg.norm(wind_velocity)
        wind_angle = np.arctan2(wind_velocity[1], wind_velocity[0]) if wind_speed > 0.01 else 0.0

        # DEFINE PHYSICAL SPEED (Meters per Second)
        # l_base represents the base rate of spread in prob/sec. 
        # 0.001 m/s = very slow creep (realističky pomalé)
        # 0.01 m/s = slow creep  
        # 0.1 m/s = moderate spread
        PHYSICAL_SPREAD_SPEED = 0.1  # Rychlejší šíření pro viditelnost
        
        # CALCULATE RATE PARAMETER (Lambda)
        # Speed = Cell_Size * Rate  =>  Rate = Speed / Cell_Size
        scaled_spread_rate = PHYSICAL_SPREAD_SPEED / cell_size_m

        # Calculate fuel properties based on cell size
        BURN_RATE_GRASS_PER_M2 = 0.01      # 30s pro 1x1m buňku
        cell_area = cell_size_m * cell_size_m  # m2
        FUEL_GRASS = (0.3, BURN_RATE_GRASS_PER_M2 / cell_area)       # Scale podle plochy

        # print(scaled_spread_rate)

        self.fire_grid = FireGrid(
            H=H, W=W, dt=dt, alpha=alpha, 
            k_wind=k_wind, k_slope=1.0,
            wind_dir=wind_angle, 
            l_base=np.ones(H) * scaled_spread_rate
        )
        self.fire_enabled = True
        
        if self._pending_terrain_data:
            # print("    Found pending terrain data. Rasterizing now...")
            self.rasterize_terrain_layers(*self._pending_terrain_data)
            self._pending_terrain_data = None
        else:
            # print("⚠️  No terrain data found. Initializing as grass with correct burn rates.")
            # Použij naše nové fuel properties místo starých konstant!
            self.fire_grid.F[:] = FUEL_GRASS[0]
            self.fire_grid.fuel_burn_rate[:] = FUEL_GRASS[1]

        self.fire_grid.B[:] = False
        self.fire_grid.I[:] = 0.0
        
        # print(f"✅ Fire simulation ready: {H}x{W} cells, {cell_size_m}m resolution.")

    def start_fire_at_position(self, world_pos, intensity=0.2, radius_m=5.0):
        """
        Založí oheň na dané pozici a v jejím okolí (vytvoří velkou počáteční skvrnu).
        radius_m=5.0 vytvoří čtverec o straně cca 10x10 metrů.
        """
        if not self.fire_enabled: return False
        try:
            # Střed ohně
            center_i, center_j = self.grid_mapper.world_to_cell(world_pos)
            
            # Kolik buněk odpovídá zadanému poloměru
            radius_cells = int(radius_m / self.grid_mapper.cell_size_m)
            
            ignited_any = False
            
            # Projdeme čtverec kolem středu
            for i in range(center_i - radius_cells, center_i + radius_cells + 1):
                for j in range(center_j - radius_cells, center_j + radius_cells + 1):
                    # Kontrola, abychom nezapisovali mimo matici
                    if 0 <= i < self.fire_grid.H and 0 <= j < self.fire_grid.W:
                        # Pokud je na políčku palivo, zapal ho
                        if self.fire_grid.F[i, j] > 0:
                            self.fire_grid.B[i, j] = True
                            # Intenzitu omezíme podle množství paliva
                            self.fire_grid.I[i, j] = np.minimum(1.0, self.fire_grid.F[i, j] * intensity)
                            ignited_any = True
                            
            if ignited_any:
                # print(f"🔥 Oheň založen na {world_pos} (blok {radius_m*2}x{radius_m*2}m)")
                return True
                
        except Exception as e:
            pass
            
        print(f"❌ Nepodařilo se založit oheň na {world_pos} (Žádné palivo / Mimo mapu)")
        return False

    def update_fire_simulation(self, suppression_assignments=None, water_drops=None, real_dt=None):
        if not self.fire_enabled: return
        
        if real_dt is None: real_dt = self.fire_grid.dt
        self.fire_time_accumulator += real_dt
        
        # Accumulate time steps to run fire physics
        # This allows fire physics to run at a different rate than PyBullet
        steps_to_run = int(self.fire_time_accumulator / self.fire_grid.dt)
        if steps_to_run > 0:
            self._update_wind_dynamics(dt=steps_to_run * self.fire_grid.dt)
            
            wind_v = self.weather['wind_velocity'][:2]
            if np.linalg.norm(wind_v) > 0.01:
                self.fire_grid.wind_dir = np.arctan2(wind_v[1], wind_v[0])
            
            # Run fire steps
            for _ in range(steps_to_run):
                self.fire_grid.step(suppression_assignments, water_drops)
                
            self.fire_time_accumulator -= steps_to_run * self.fire_grid.dt

    def get_fire_state(self):
        if not self.fire_enabled: return None
        return {
            'fire_grid_state': self.fire_grid.get_state(),
            'fire_stats': self.fire_grid.get_stats(),
            'grid_bounds': self.grid_mapper.get_grid_bounds(),
            'cell_size': self.grid_mapper.cell_size_m
        }

    # ============================================================================
    # VISUALIZATION UTILS
    # ============================================================================
    
    def visualize_fire_in_simulation(self):
        if not self.fire_enabled: return
        
        for obj_id in self.fire_visual_objects:
            p.removeBody(obj_id)
        self.fire_visual_objects.clear()
        
        burning_indices = np.argwhere(self.fire_grid.B)
        # Downsample for visualization performance if too many cells
        if len(burning_indices) > 500:
            indices_to_show = burning_indices[::int(len(burning_indices)/500)]
        else:
            indices_to_show = burning_indices
            
        for i, j in indices_to_show:
            world_pos = self.grid_mapper.cell_to_world(i, j)
            intensity = self.fire_grid.I[i, j]
            h = 0.5 + intensity * 2.0
            rad = self.grid_mapper.cell_size_m * 0.4
            
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=rad, length=h, rgbaColor=[1, 0, 0, 0.8])
            obj = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[world_pos[0], world_pos[1], h/2])
            self.fire_visual_objects.append(obj)

    def save_environment_map(self, filename="environment_map.png", show_fire_grid=True, detailed=False):
        """
        Save a top-down visualization of the environment.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        print(f"📸 Saving environment map to {filename}...")
        
        if self.fire_grid is not None:
            # Use FireGrid bounds
            x_min, x_max, y_min, y_max = self.grid_mapper.get_grid_bounds()
            
            # Create RGB image based on fuel type
            H, W = self.fire_grid.H, self.fire_grid.W
            fuel_map = self.fire_grid.F
            
            # Colors: 
            # Grass (0.3) = Light Green
            # Forest (0.8) = Dark Green
            # Building (0.9) = Gray
            # Water (0.0) = Blue
            
            img = np.zeros((H, W, 3))
            
            # Grass base
            img[:, :] = [0.8, 0.9, 0.6] 
            
            # Forest
            mask_forest = (fuel_map > 0.75) & (fuel_map < 0.85)
            img[mask_forest] = [0.1, 0.4, 0.1]
            
            # Buildings
            mask_buildings = (fuel_map > 0.85)
            img[mask_buildings] = [0.5, 0.5, 0.5]
            
            # Water
            mask_water = (fuel_map < 0.05)
            img[mask_water] = [0.2, 0.5, 0.9]
            
            plt.figure(figsize=(10, 10))
            plt.imshow(img, extent=[x_min, x_max, y_min, y_max], origin='lower')
            plt.title("Fire Grid Fuel Map (Rasterized)")
            plt.xlabel("X (meters)")
            plt.ylabel("Y (meters)")
            
            # Legend
            patches = [
                mpatches.Patch(color=[0.8, 0.9, 0.6], label='Grass (Low Fuel)'),
                mpatches.Patch(color=[0.1, 0.4, 0.1], label='Forest (High Fuel)'),
                mpatches.Patch(color=[0.5, 0.5, 0.5], label='Building (Slow Burn)'),
                mpatches.Patch(color=[0.2, 0.5, 0.9], label='Water (Firebreak)')
            ]
            plt.legend(handles=patches, loc='upper right')
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            # print(f"✅ Saved map: {filename}")
        else:
            print("❌ Cannot save map: FireGrid not initialized")

    def create_refill_zone(self, center_pos=None, size=30.0):
        """
        Creates a visual refill zone.
        If center_pos is None, generates a random position.
        """
        # Pokud není zadána pozice, vygenerujeme náhodnou (např. v rozsahu +/- 200m)
        if center_pos is None:
             x = random.uniform(-100, 100)
             y = random.uniform(-100, 100)
             z = random.uniform(30, 80) # Výška vhodná pro letadla
             center_pos = [x, y, z]

        # Vytvoření vizuálu (Modrá, poloprůhledná krychle)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[size/2, size/2, size/2], # 5x5x5 metrů
            rgbaColor=[0, 1, 1, 0.4] # Cyan, 40% průhlednost
        )
        
        # Vytvoření tělesa bez kolizí (baseCollisionShapeIndex=-1)
        zone_id = p.createMultiBody(
            baseVisualShapeIndex=visual_shape,
            basePosition=center_pos
        )
        
        self.refill_zone = {
            'id': zone_id,
            'position': np.array(center_pos),
            'size': size,
            'radius_sq': (size/2 + 2.0)**2 # Čtverec poloměru pro rychlou detekci (+ tolerance)
        }
        
        # print(f"💧 Refill Zone created at {center_pos}")
        return center_pos
    
    # ============================================================================
    # WIND & PHYSICS UTILS
    # ============================================================================

    def _initialize_random_wind(self):
        speed = random.uniform(3.0, 25.0)
        angle = random.uniform(0, 2 * np.pi)
        self.wind_velocity = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])

    def _update_wind_dynamics(self, dt):
        self.wind_change_timer += dt
        if self.wind_change_timer >= self.wind_change_interval:
            # Perturb wind slightly
            current_speed = np.linalg.norm(self.target_wind[:2])
            current_angle = np.arctan2(self.target_wind[1], self.target_wind[0])
            
            new_speed = np.clip(current_speed + random.uniform(-1, 1), 2.0, 15.0)
            new_angle = current_angle + random.uniform(-0.2, 0.2)
            
            self.target_wind = np.array([new_speed * np.cos(new_angle), new_speed * np.sin(new_angle), 0.0])
            self.wind_change_timer = 0.0
            self.wind_change_interval = random.uniform(5.0, 15.0)
            
        # Smooth transition
        self.wind_velocity = 0.98 * self.wind_velocity + 0.02 * self.target_wind
        self.weather['wind_velocity'] = self.wind_velocity

    def set_wind(self, wind_velocity):
        """
        Manually set wind velocity and disable random updates.
        
        Args:
            wind_velocity: [vx, vy, vz] in m/s
        """
        self.wind_velocity = np.array(wind_velocity, dtype=float)
        self.target_wind = self.wind_velocity.copy()
        self.weather['wind_velocity'] = self.wind_velocity
        
        # Disable random wind changes
        self.manual_wind_control = True
        # print(f"🌬️ Environment Wind set to: {self.wind_velocity} m/s")

    def get_wind_at_position(self, position):
        return self.wind_velocity * (1.0 + position[2] * 0.01)

    def is_position_in_obstacle(self, position):
        for obstacle in self.obstacles:
            b = obstacle['bounds']
            if (b['min'][0] <= position[0] <= b['max'][0] and
                b['min'][1] <= position[1] <= b['max'][1] and
                b['min'][2] <= position[2] <= b['max'][2]):
                return True, obstacle
        return False, None

    def _calculate_bounds(self, position, size):
        return {
            'min': [position[0] - size[0]/2, position[1] - size[1]/2, 0],
            'max': [position[0] + size[0]/2, position[1] + size[1]/2, size[2]]
        }
    
    def get_environment_info(self):
        return {
            'obstacles': len(self.obstacles),
            'terrain_zones': len(self.terrain_zones),
            'weather': self.weather.copy()
        }