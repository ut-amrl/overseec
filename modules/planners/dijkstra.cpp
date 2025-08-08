#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <queue>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <pybind11/iostream.h>
#include <cstdio>  // For fprintf, fflush

namespace py = pybind11;
using namespace std;

#define IDX(y, x) ((y) * W + (x))

struct Node {
    int x, y;
    double cost;
    bool operator>(const Node& other) const {
        return cost > other.cost;
    }
};

// ========== 1. Dijkstra Path ==========
py::list dijkstra_path(py::array_t<double> costmap, int sx, int sy, int gx, int gy) {
    auto buf = costmap.request();
    int H = buf.shape[0], W = buf.shape[1];
    auto ptr = static_cast<double*>(buf.ptr);

    vector<double> dist(H * W, 1e9);
    vector<int> parent_x(H * W, -1);
    vector<int> parent_y(H * W, -1);
    vector<uint8_t> visited(H * W, 0);

    auto at = [&](int y, int x) -> double& {
        return *(ptr + IDX(y, x));
    };

    priority_queue<Node, vector<Node>, greater<Node>> open;
    dist[IDX(sy, sx)] = 0.0;
    open.push({sx, sy, 0.0});

    while (!open.empty()) {
        Node curr = open.top(); open.pop();
        int x = curr.x, y = curr.y;
        if (visited[IDX(y, x)]) continue;
        visited[IDX(y, x)] = 1;

        if (x == gx && y == gy) break;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H || visited[IDX(ny, nx)]) continue;

                double step = at(ny, nx);
                if (dx != 0 && dy != 0) step *= 1.4142135;
                double new_cost = dist[IDX(y, x)] + step;

                if (new_cost < dist[IDX(ny, nx)]) {
                    dist[IDX(ny, nx)] = new_cost;
                    parent_x[IDX(ny, nx)] = x;
                    parent_y[IDX(ny, nx)] = y;
                    open.push({nx, ny, new_cost});
                }
            }
        }
    }

    py::list path;
    int x = gx, y = gy;
    if (parent_x[IDX(y, x)] == -1 && !(x == sx && y == sy)) return path;

    while (true) {
        path.insert(0, py::make_tuple(x, y));
        if (x == sx && y == sy) break;
        int px = parent_x[IDX(y, x)];
        int py_ = parent_y[IDX(y, x)];
        x = px; y = py_;
    }

    return path;
}

// ========== 2. Value at Pixel ==========
double value_at_pixel(
    py::array_t<double> costmap,
    py::array_t<int> semantic_map,
    py::dict rank_dict,
    int x0, int y0,
    int gx, int gy,
    double beta
) {
    py::list path = dijkstra_path(costmap, x0, y0, gx, gy);
    if (path.size() == 0) return 0.0;

    auto sem_buf = semantic_map.request();
    int H = sem_buf.shape[0], W = sem_buf.shape[1];
    auto sem_ptr = static_cast<int*>(sem_buf.ptr);

    unordered_map<int, int> rank_map;
    for (auto item : rank_dict) {
        int cls = py::cast<int>(item.first);
        int rank = py::cast<int>(item.second);
        rank_map[cls] = rank;
    }

    double total = 0.0;
    int count = 0;

    for (auto p : path) {
        auto xy = py::cast<py::tuple>(p);
        int x = xy[0].cast<int>();
        int y = xy[1].cast<int>();

        int cls = sem_ptr[IDX(y, x)];
        if (rank_map.count(cls)) {
            int rank = rank_map[cls];
            total += pow(beta, rank - 1);
            count++;
        }
    }

    return (count > 0) ? (total / count) : 0.0;
}

// ========== 3. Value of Trajectory ==========
double evaluate_trajectory(
    py::array_t<double> costmap,
    py::array_t<int> semantic_map,
    py::dict rank_dict,
    py::list trajectory_points,
    int gx, int gy,
    double beta
) {
    double total_value = 0.0;
    int count = 0;
    int total = trajectory_points.size();
    int report_every = std::max(1, total / 50);  // update every 2%

    for (int i = 0; i < total; ++i) {
        auto xy = py::cast<py::tuple>(trajectory_points[i]);
        int x = xy[0].cast<int>();
        int y = xy[1].cast<int>();
        double v = value_at_pixel(costmap, semantic_map, rank_dict, x, y, gx, gy, beta);
        total_value += v;
        count++;

        // if (i % report_every == 0 || i == total - 1) {
        fprintf(stderr, "\r[Trajectory] Progress: %3d / %3d", count, total);
            // fflush(stderr);
        // }
    }

    fprintf(stderr, "\n");  // New line after progress
    return (count > 0) ? (total_value / count) : 0.0;
}

// ========== 4. Dijkstra Tree From Goal with Progress ==========
py::array_t<int> dijkstra_tree_from_goal(py::array_t<double> costmap, int gx, int gy) {
    auto buf = costmap.request();
    int H = buf.shape[0], W = buf.shape[1];
    auto ptr = static_cast<double*>(buf.ptr);

    vector<double> dist(H * W, 1e9);
    vector<int> parent_x(H * W, -1);
    vector<int> parent_y(H * W, -1);
    vector<uint8_t> visited(H * W, 0);

    auto at = [&](int y, int x) -> double& {
        return *(ptr + IDX(y, x));
    };

    int total_pixels = H * W;
    int visited_count = 0;
    int report_every = std::max(1, total_pixels / 100);  // print every 1%

    priority_queue<Node, vector<Node>, greater<Node>> open;
    dist[IDX(gy, gx)] = 0.0;
    open.push({gx, gy, 0.0});

    while (!open.empty()) {
        Node curr = open.top(); open.pop();
        int x = curr.x, y = curr.y;
        if (visited[IDX(y, x)]) continue;
        visited[IDX(y, x)] = 1;
        visited_count++;

        if (visited_count % report_every == 0 || visited_count == total_pixels) {
            fprintf(stderr, "\r[Dijkstra Tree] Visited %d / %d (%.1f%%)", visited_count, total_pixels,
                    (100.0 * visited_count) / total_pixels);
            fflush(stderr);
        }

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H || visited[IDX(ny, nx)]) continue;

                double step = at(ny, nx) + 1.0;
                if (dx != 0 && dy != 0) step *= 1.4142135;
                double new_cost = dist[IDX(y, x)] + step;

                if (new_cost < dist[IDX(ny, nx)]) {
                    dist[IDX(ny, nx)] = new_cost;
                    parent_x[IDX(ny, nx)] = x;
                    parent_y[IDX(ny, nx)] = y;
                    open.push({nx, ny, new_cost});
                }
            }
        }
    }

    fprintf(stderr, "\n");  // finish with newline

    // Prepare output array of shape [H, W, 2]
    auto result = py::array_t<int>({H, W, 2});
    auto r = result.mutable_unchecked<3>();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            r(y, x, 0) = parent_x[IDX(y, x)];
            r(y, x, 1) = parent_y[IDX(y, x)];
        }
    }

    return result;
}

// ========== pybind11 Bindings ==========
PYBIND11_MODULE(dijkstra_bind, m){
    m.def("dijkstra_path", &dijkstra_path, "Returns optimal path using Dijkstra");
    m.def("value_at_pixel", &value_at_pixel, "Returns semantic value from pixel to goal");
    m.def("evaluate_trajectory", &evaluate_trajectory, "Returns mean value of a trajectory");
    m.def("dijkstra_tree_from_goal", &dijkstra_tree_from_goal, "Returns Dijkstra tree from goal");
}
