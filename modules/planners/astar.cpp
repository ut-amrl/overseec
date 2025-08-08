#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <queue>
#include <vector>
#include <cmath>

namespace py = pybind11;

using namespace std;

struct Node {
    int x, y;
    double f;
    // The operator> is correctly defined for a min-priority queue.
    bool operator>(const Node& other) const {
        return f > other.f;
    }
};

py::list astar(py::array_t<double> costmap, int sx, int sy, int gx, int gy) {
    auto buf = costmap.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Costmap must be a 2D array");
    }
    int H = buf.shape[0], W = buf.shape[1];
    auto ptr = static_cast<double*>(buf.ptr);

    // Lambda to access costmap elements safely.
    auto at = [&](int y, int x) -> double& {
        return *(ptr + y * W + x);
    };

    vector<vector<double>> g(H, vector<double>(W, 1e9));
    vector<vector<int>> parent_x(H, vector<int>(W, -1));
    vector<vector<int>> parent_y(H, vector<int>(W, -1));
    vector<vector<bool>> visited(H, vector<bool>(W, false));

    priority_queue<Node, vector<Node>, greater<Node>> open;
    
    g[sy][sx] = 0;
    double h_start = hypot(gx - sx, gy - sy);
    open.push({sx, sy, h_start});

    while (!open.empty()) {
        Node curr = open.top(); open.pop();
        int x = curr.x, y = curr.y;

        if (visited[y][x]) {
            continue;
        }
        visited[y][x] = true;

        if (x == gx && y == gy) {
            break; // Goal reached
        }

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;
                
                int nx = x + dx, ny = y + dy;

                if (nx < 0 || nx >= W || ny < 0 || ny >= H || visited[ny][nx]) {
                    continue;
                }

                // --- CHANGE #1: Refactored Cost Calculation (Best Practice) ---
                // The cost of a step is separated into the cost to enter the node
                // and the cost of the movement itself (distance-based).
                double move_dist = (dx != 0 && dy != 0) ? 1.41421356237 : 1.0;
                double base_move_cost = 4.0; // This is your base movement cost.
                
                // Cost to move from (x,y) to (nx,ny)
                double step_cost = base_move_cost * move_dist;
                // Cost associated with entering the destination node
                double node_cost = at(ny, nx);
                
                double tg = g[y][x] + step_cost + node_cost;

                if (tg < g[ny][nx]) {
                    g[ny][nx] = tg;
                    parent_x[ny][nx] = x;
                    parent_y[ny][nx] = y;
                    double h = hypot(gx - nx, gy - ny);
                    
                    // --- CHANGE #2: Heuristic Tie-Breaker ---
                    // Multiply the heuristic by a small factor > 1. This makes the
                    // search greedier, breaking ties in favor of nodes closer
                    // to the goal and ensuring a straighter path.
                    double tie_breaker = 1.1;
                    open.push({nx, ny, tg + h * tie_breaker});
                }
            }
        }
    }

    // Reconstruct path
    py::list path;
    int x = gx, y = gy;
    // Check if a path was found.
    if (parent_x[y][x] == -1 && !(x == sx && y == sy)) {
        return path; // Return empty list if no path
    }

    while (true) {
        path.insert(0, py::make_tuple(x, y));
        if (x == sx && y == sy) {
            break;
        }
        int px = parent_x[y][x];
        int py_ = parent_y[y][x];
        x = px;
        y = py_;
    }

    return path;
}

PYBIND11_MODULE(astar_bind, m) {
    m.def("astar", &astar, "Run A* on a 2D costmap",
          py::arg("costmap"), py::arg("sx"), py::arg("sy"), py::arg("gx"), py::arg("gy"));
}
