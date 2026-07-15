#include "bfs_distance.h"
#include <cstring>
#include <vector>

// Persistent scratch queue to avoid per-call heap allocation
static std::vector<int> g_queue;
static int g_alloc_size = 0;

py::array_t<int16_t> bfs_distance_grid(
    int width, int height,
    py::array passable_mask,
    py::array shelf_mask,
    int target_x, int target_y)
{
    const int N = height * width;

    // Allocate output numpy array
    py::array_t<int16_t> result({height, width});
    int16_t* dist = static_cast<int16_t*>(result.request().ptr);

    // Initialize all to -1 (0xFFFF for int16_t)
    std::memset(dist, 0xFF, N * sizeof(int16_t));

    // Validate target bounds
    if (target_x < 0 || target_x >= width || target_y < 0 || target_y >= height) {
        return result;
    }

    // Get raw pointers — accept bool or uint8 (both are 1-byte, truthy != 0)
    auto pbuf = passable_mask.request();
    auto sbuf = shelf_mask.request();
    const uint8_t* passable = static_cast<const uint8_t*>(pbuf.ptr);
    const uint8_t* shelf    = static_cast<const uint8_t*>(sbuf.ptr);

    const int target_idx = target_y * width + target_x;
    if (!passable[target_idx]) {
        return result;
    }

    // Ensure queue scratch is large enough
    if (N > g_alloc_size) {
        g_queue.resize(static_cast<size_t>(N));
        g_alloc_size = N;
    }

    // BFS from target outward (reversed: distance = steps to reach target)
    dist[target_idx] = 0;
    int head = 0, tail = 0;
    g_queue[tail++] = target_idx;

    static constexpr int dx[4] = { 0,  0, -1,  1};
    static constexpr int dy[4] = {-1,  1,  0,  0};

    while (head < tail) {
        const int idx  = g_queue[head++];
        const int x    = idx % width;
        const int y    = idx / width;
        const int16_t d = dist[idx];
        const bool cur_is_shelf = (shelf[idx] != 0);

        for (int i = 0; i < 4; ++i) {
            const int nx = x + dx[i];
            const int ny = y + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const int nidx = ny * width + nx;
            if (dist[nidx] != -1) continue;       // already visited
            if (!passable[nidx]) continue;         // impassable
            // Shelf-to-shelf restriction: cannot move shelf <-> shelf
            if (cur_is_shelf && shelf[nidx]) continue;

            dist[nidx] = d + 1;
            g_queue[tail++] = nidx;
        }
    }

    return result;
}
