// Demo configuration
// Add or remove entries from the `demos` array to control which demos appear on the website.
// Each entry should have:
//   id:            unique identifier (used for HTML element IDs)
//   label:         display name shown above the demo
//   folder:        path to the demo folder (relative to website root)
//   rgb:           filename of the RGB image (JPG/PNG)
//   costmap:       filename of the costmap image for display (PNG)
//   costmap_gray:  filename of the grayscale costmap for A* planner (PNG, single-channel)
//
// A* planner config (matches astar.cpp variable names):
//   base_move_cost:  base cost per unit distance of movement (default: 4.0)
//   tie_breaker:     heuristic multiplier for tie-breaking (default: 1.1)
//   path_color:      color of the planned path line (default: "#00ffff")
//   path_width:      width of the planned path line in pixels (default: 6)

const DEMO_CONFIG = {
  // Global A* defaults (can be overridden per-demo)
  planner: {
    base_move_cost: 4.0,
    tie_breaker: 1.1,
    path_color: "#00ffff",
    path_width: 6,
  },

  demos: [
    {
      id: "pickle_south",
      label: "Pickle South",
      folder: "images/demo/pickle_south",
      rgb: "rgb.jpg",
      costmap: "costmap.png",
      costmap_gray: "costmap_gray.png",
    },

    {
      id: "onion_creek",
      label: "Onion Creek",
      folder: "images/demo/onion_creek",
      rgb: "rgb.jpg",
      costmap: "costmap.png",
      costmap_gray: "costmap_gray.png",
    },
  ],
};
