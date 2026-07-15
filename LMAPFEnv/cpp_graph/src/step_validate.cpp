#include "step_validate.h"

py::array_t<uint8_t> validate_agent_actions(
    int width, int height,
    py::array_t<int32_t> agv_x,
    py::array_t<int32_t> agv_y,
    py::array_t<int32_t> actions,
    py::array passable,
    py::array shelf)
{
    auto buf_x = agv_x.request();
    auto buf_y = agv_y.request();
    auto buf_a = actions.request();
    auto buf_p = passable.request();
    auto buf_s = shelf.request();

    const int32_t* x_ptr = static_cast<const int32_t*>(buf_x.ptr);
    const int32_t* y_ptr = static_cast<const int32_t*>(buf_y.ptr);
    const int32_t* a_ptr = static_cast<const int32_t*>(buf_a.ptr);
    const uint8_t* p_ptr = static_cast<const uint8_t*>(buf_p.ptr);
    const uint8_t* s_ptr = static_cast<const uint8_t*>(buf_s.ptr);

    const int n = static_cast<int>(buf_x.size);

    // Allocate output array
    py::array_t<uint8_t> result(n);
    auto buf_r = result.request();
    uint8_t* r_ptr = static_cast<uint8_t*>(buf_r.ptr);

    // Validate each agent's action
    for (int i = 0; i < n; ++i) {
        const int x = x_ptr[i];
        const int y = y_ptr[i];
        const int action = a_ptr[i];

        // STAY action is always feasible
        if (action == 4) {
            r_ptr[i] = 1;
            continue;
        }

        // Compute target position
        int tx = x, ty = y;
        if (action == 0) ty = y - 1;       // UP
        else if (action == 1) ty = y + 1;  // DOWN
        else if (action == 2) tx = x - 1;  // LEFT
        else if (action == 3) tx = x + 1;  // RIGHT

        // Check bounds
        if (tx < 0 || tx >= width || ty < 0 || ty >= height) {
            r_ptr[i] = 0;
            continue;
        }

        const int cur_idx = y * width + x;
        const int tgt_idx = ty * width + tx;

        // Check passability
        if (!p_ptr[tgt_idx]) {
            r_ptr[i] = 0;
            continue;
        }

        // Check shelf-to-shelf restriction
        if (s_ptr[cur_idx] && s_ptr[tgt_idx]) {
            r_ptr[i] = 0;
            continue;
        }

        r_ptr[i] = 1;
    }

    return result;
}
