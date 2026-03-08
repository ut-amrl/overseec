// Demo configuration
// Add or remove entries from the `demos` array to control which demos appear on the website.
// Each entry should have:
//   id:      unique identifier (used for HTML element IDs)
//   label:   display name shown above the demo
//   folder:  path to the demo folder (relative to website root)
//   rgb:     filename of the RGB image (JPG/PNG)
//   prompts: array of prompt entries, each with:
//     label:         natural language prompt that generated this costmap
//     costmap:       filename of the costmap image for display (PNG)
//     costmap_gray:  filename of the grayscale costmap for the planner (PNG, single-channel)
//
// Planner config (matches astar.cpp variable names):
//   base_move_cost:  base cost per unit distance of movement (default: 4.0)
//   tie_breaker:     heuristic multiplier for tie-breaking (default: 1.1)
//   path_color:      color of the planned path line (default: "#00ffff")
//   path_width:      width of the planned path line in pixels (default: 6)

const DEMO_CONFIG = {
  // Global planner defaults (can be overridden per-demo)
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
      prompts: [
        {
          label: "I want to stay on road. Everything else is lethal",
          costmap: "costmap.png",
          costmap_gray: "costmap_gray.png",
        },
        {
          label: "I prefer the road and grass. Avoid everything else.",
          costmap: "costmap_2.png",
          costmap_gray: "costmap_2_gray.png",
        },
      ],
    },

    {
      id: "onion_creek",
      label: "Onion Creek",
      folder: "images/demo/onion_creek",
      rgb: "rgb.jpg",
      prompts: [
        {
          label: "I want to stay on only trails. Everything else is lethal. Road is also lethal",
          costmap: "costmap.png",
          costmap_gray: "costmap_gray.png",
        },
      ],
    },
  ],
};
