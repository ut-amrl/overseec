// A* planner — direct port of modules/planners/astar.cpp
// Uses the same variable names and cost model.

function astar(costmap, H, W, sx, sy, gx, gy, base_move_cost, tie_breaker) {
  // costmap: Float64Array or similar, row-major [H x W]
  // Returns array of [x, y] pairs, or empty array if no path.

  function at(y, x) {
    return costmap[y * W + x];
  }

  // g-values, parents, visited
  const g = new Float64Array(H * W).fill(Infinity);
  const parent_x = new Int32Array(H * W).fill(-1);
  const parent_y = new Int32Array(H * W).fill(-1);
  const visited = new Uint8Array(H * W);

  // Min-heap using a binary heap (priority queue)
  // Each entry: { x, y, f }
  const open = [];

  function heapPush(node) {
    open.push(node);
    let i = open.length - 1;
    while (i > 0) {
      const pi = (i - 1) >> 1;
      if (open[pi].f > open[i].f) {
        [open[pi], open[i]] = [open[i], open[pi]];
        i = pi;
      } else break;
    }
  }

  function heapPop() {
    const top = open[0];
    const last = open.pop();
    if (open.length > 0) {
      open[0] = last;
      let i = 0;
      while (true) {
        let smallest = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < open.length && open[l].f < open[smallest].f) smallest = l;
        if (r < open.length && open[r].f < open[smallest].f) smallest = r;
        if (smallest !== i) {
          [open[i], open[smallest]] = [open[smallest], open[i]];
          i = smallest;
        } else break;
      }
    }
    return top;
  }

  const idx = (y, x) => y * W + x;

  g[idx(sy, sx)] = 0;
  const h_start = Math.hypot(gx - sx, gy - sy);
  heapPush({ x: sx, y: sy, f: h_start });

  while (open.length > 0) {
    const curr = heapPop();
    const x = curr.x, y = curr.y;

    if (visited[idx(y, x)]) continue;
    visited[idx(y, x)] = 1;

    if (x === gx && y === gy) break;

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        if (dx === 0 && dy === 0) continue;

        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= W || ny < 0 || ny >= H || visited[idx(ny, nx)]) continue;

        const move_dist = (dx !== 0 && dy !== 0) ? 1.41421356237 : 1.0;
        const step_cost = base_move_cost * move_dist;
        const node_cost = at(ny, nx);

        const tg = g[idx(y, x)] + step_cost + node_cost;

        if (tg < g[idx(ny, nx)]) {
          g[idx(ny, nx)] = tg;
          parent_x[idx(ny, nx)] = x;
          parent_y[idx(ny, nx)] = y;
          const h = Math.hypot(gx - nx, gy - ny);
          heapPush({ x: nx, y: ny, f: tg + h * tie_breaker });
        }
      }
    }
  }

  // Reconstruct path
  const path = [];
  let x = gx, y = gy;
  if (parent_x[idx(y, x)] === -1 && !(x === sx && y === sy)) {
    return path; // No path found
  }

  while (true) {
    path.unshift([x, y]);
    if (x === sx && y === sy) break;
    const px = parent_x[idx(y, x)];
    const py_ = parent_y[idx(y, x)];
    x = px;
    y = py_;
  }

  return smoothPath(path, costmap, H, W);
}

// Line-of-sight path smoothing using Bresenham ray checks.
// Removes unnecessary waypoints where a straight line stays on
// similarly-costed cells, eliminating grid-artifact zigzags.
function smoothPath(path, costmap, H, W) {
  if (path.length <= 2) return path;

  // Check if a straight line from (x0,y0) to (x1,y1) is collision-free.
  // "Collision" = any cell along the line has cost > max_cost_threshold,
  // where the threshold is the max cost of the two endpoints * a tolerance.
  function lineOfSight(x0, y0, x1, y1) {
    const endCost = Math.max(costmap[y0 * W + x0], costmap[y1 * W + x1]);
    const threshold = endCost + 50; // allow some tolerance

    let dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0);
    let sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;
    let cx = x0, cy = y0;

    while (true) {
      if (cx < 0 || cx >= W || cy < 0 || cy >= H) return false;
      if (costmap[cy * W + cx] > threshold) return false;
      if (cx === x1 && cy === y1) break;
      const e2 = 2 * err;
      if (e2 > -dy) { err -= dy; cx += sx; }
      if (e2 <  dx) { err += dx; cy += sy; }
    }
    return true;
  }

  const smoothed = [path[0]];
  let i = 0;
  while (i < path.length - 1) {
    // Greedily skip as far ahead as line-of-sight allows
    let farthest = i + 1;
    for (let j = path.length - 1; j > i + 1; j--) {
      if (lineOfSight(path[i][0], path[i][1], path[j][0], path[j][1])) {
        farthest = j;
        break;
      }
    }
    smoothed.push(path[farthest]);
    i = farthest;
  }
  return smoothed;
}
