import ast
import requests
import textwrap

SERVER_URL = "http://0.0.0.0:8057/generate"  # Update if running on a different host

def extract_dict_and_write_code(text: str, filepath: str):
    # Extract dictionary
    dict_start = text.find("<DICT>")
    dict_end = text.find("</DICT>")
    if dict_start == -1 or dict_end == -1:
        raise ValueError("Missing <DICT> tags")
    dict_str = text[dict_start + 6:dict_end].strip()
    parsed_dict = ast.literal_eval(dict_str)

    # Extract code
    code_start = text.find("<CODE>")
    code_end = text.find("</CODE>")
    if code_start == -1 or code_end == -1:
        raise ValueError("Missing <CODE> tags")
    code_str = text[code_start + 6:code_end].strip()

    # Write code to file
    with open(filepath, 'w') as f:
        f.write(code_str + '\n')

    print(parsed_dict)
    return parsed_dict

def wrap_prompt(prompt: str) -> str:
    return f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

# def overseec_query_llm(user_prompt, code_filepath):

#     prompt = """

#         example prefernce : "avoid the baseball field. the pond is dry, so you can go over it. stay on the sides of the road, dont go over water. "

#         you have 3 tasks:

#         TASK 1 : class segregation
#             Thresholds:
#             TL = 0.4,
#             TA = 0.8

#             From this prompt, extract geographic classes, semantic classes and entities (e.g., road, grass, tree, building, water, trails, bushes, river, etc.).
#             It can be an areal feature, a linear feature, an object, or a subset of a larger class.

#             Use descriptive names if adjectives are provided (e.g., "big trees", "curved roads").

#             Do not include specific regions like "center of x", "side of y".


#             Always include the default classes:
#             "road", "trail or footway", "water", "grass", "building", and "tree".

#             For every class, determine if it is:
#             - ‘linear/network-like’ (e.g., roads, trails): assign TL
#             - ‘areal/blob-like’ (e.g., grass, forest, buildings): assign TA

#             Output only the dictionary mapping class names to a list of assigned [threshold, <RGB color>], strictly between the <DICT></DICT> markers.

#             example "remain on the outskirts of the forest" [we did not add `outskirt of forest` as it is a specific region, but we added `forest`]

#             <DICT>{
#                 "road": [0.4, [128, 128, 128]],
#                 "trail or footway": [0.4, [160, 160, 160]],
#                 "water": [0.8, [0, 0, 255]],
#                 "grass": [0.8, [0, 255, 0]],
#                 "building": [0.8, [255, 0, 0]],
#                 "tree": [0.8, [34, 139, 34]],
#                 "baseball field": [0.8, [0, 100, 0]],
#                 "pond": [0.8, [0, 100, 255]],
#                 "farm": [0.8, [255, 255, 0]],

#             }</DICT>

#         TASK 2 :
#         Write down all the heirarchical relationships in the form of a list of tuples. Each tuple should be (parent, child).
#         parent is a broad class, child is a more specific instance. Make sure you use only these heirarchy in the code part.
#         <HIER>
#             [('water', 'pond'),
#              ('grass', 'baseball field')]
#         </HIER>

#         TASK 3 :

#         Inputs
#         - mask_dict: dict mapping lowercase class names → equally sized binary masks (NumPy np.uint8 or PyTorch bool/uint8/float on any device).
#         - t_dict: dict with thresholds { "t_l": TL, "t_a": TA } for linear vs areal classes.
#         - Helper mask ops are available: mask_and, mask_or, mask_not, and mask_remove(A, B) (i.e., A ∧ ¬B).
#         - You may use OpenCV (cv2) for morphology/DT or pure NumPy/PyTorch; autodetect types and stay in one backend.

#         Default semantics (when prompt is silent)
#         - Costs before normalization:
#         road=0, trail or footway=0, grass=100, tree=1000, building=1000, water=1000.

#                     Analyze the user's prompt below to understand their preferences. Create a "Costing Plan" that ranks all geographic classes into one of the following tiers. Explain your reasoning.

#         * **Tier 1: Preferred (Cost 0-200)**: The most desired terrain. Use for "stay on", "prefer", etc.
#         * **Tier 2: Tolerated (Cost 200-50)**: Acceptable, but not ideal. Use for "you can go over", "is fine", etc.
#         * **Tier 3: Discouraged (Cost 500-750)**: Use for phrases like, "...if you have to" or "...if road isn't available."
#         * **Tier 4: Lethal (Cost 750-900)**: last resort. Use for "avoid", "don't prefer", etc.
#         * **Tier 4: Lethal (Cost 900-1000)**: Must be avoided. Use for "don't go on", etc.

#         Goal
#         Synthesize an executable function f_LLM that translates the user’s NL prompt into a normalized global costmap C ∈ [0, 1000]^(H×W) by following steps (i)–(vi) from the paper.

#         ---

#         Procedure

#         (i) Function signature
#         def generate_costmap(mask_dict, t_dict):
#             # returns costmap with shape H×W, dtype float32, range [0, 1000]

#         (ii) Mask operators
#         Use AND, OR, NOT, REMOVE for pixelwise transforms. Implement/consume:
#         - mask_and(A,B), mask_or(A,B), mask_not(A), mask_remove(A,B) = A ∧ ¬B.

#         (iii) Prompt analysis → weights, hierarchies, geometry
#         - Produce per-class weights w_c ∈ [0,1] reflecting preference (lower = more preferred).
#         - Map hints like “stay on / prefer / can go over / avoid / don’t go on” into a weight scale (e.g., 0.0, 0.25, 0.5, 0.85, 1.0).
#         - Infer semantic hierarchies H: specific classes that are a type of a more general class. parent is a broad class, child is a more specific instance. Use hierarchy from Task 2.
#         - Infer geometric cues γ_c (e.g., “stay on the sides of the road”) that require distance fields, dilations, or band masks.

#         (iv) Mask operations (hierarchy + geometry)
#         1. Thresholding (use TL for linear/network-like, TA for areal/blob-like):
#         road, trail -> TL
#         grass, tree, building, water, other areal -> TA
#         2. Hierarchy enforcement: remove child masks from parents to prevent double counting:
#         parent_mask = mask_remove(parent_mask, child_mask) for all (parent, child) ∈ H.
#         3. Geometry transforms: for each cue in γ_c, derive auxiliary masks (e.g., distance-to-centerline, bands near road edges via distanceTransform or morphological ops) and incorporate them when computing the class contribution (see v).

#         (v) Cost accumulation (per-class contributions, then sum)
#         For each class c with probability/logit map P_c (after thresholding → binary mask M_c):
#         - Convert any logits to probabilities if needed (sigmoid on float logits; if already binary, cast to float).
#         - Compute per-class cost contribution:
#         C_c = α_c * w_c * M_c * P_c * G_c
#         where:
#         - α_c is a base class cost from defaults above (adapted by the prompt when specified).
#         - G_c is an optional geometric factor in [0,1] (e.g., band near road gets lower factor if “prefer sides”; centerline gets higher factor if “avoid center”).
#         - Sum all C_c pixelwise to get C̃.
#         - Unknown region handling: pixels with no class evidence receive a high fallback (e.g., max(C̃)) to discourage exploration outside known data.

#         (vi) Normalization
#         - Normalize C̃ to [0,1] within the image (robust min–max; guard against all-zeros).
#         - Scale to [0,1000] → final C. Ensure dtype float32.

#         ---

#         def generate_costmap(mask_dict, t_dict):
#             # mask operations

#             return costmap

#         example of a costmap is :
#         example prefernce : "dont go over the baseball field. the pond is dry, so you can go over it."

#         Example implementation

#         <CODE>
#         import numpy as np
#         import cv2
#         import torch

#         def mask_and(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
#             return torch.logical_and(mask1.bool(), mask2.bool()).to(torch.uint8)

#         def mask_or(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
#             return torch.logical_or(mask1.bool(), mask2.bool()).to(torch.uint8)

#         def mask_not(mask: torch.Tensor) -> torch.Tensor:
#             return torch.logical_not(mask.bool()).to(torch.uint8)

#         def mask_remove(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
#             return (mask1.bool() & ~mask2.bool()).to(torch.uint8)


#         def generate_costmap(mask_dict, t_dict={"t_l":0.4, "t_a":0.6}):
#             shape = next(iter(mask_dict.values())).shape

#             device = next(iter(mask_dict.values())).device
#             road_logit = mask_dict.get('road', torch.zeros(shape, dtype=torch.float32, device=device))
#             trees_logit = mask_dict.get('tree', torch.zeros(shape, dtype=torch.float32, device=device))
#             buildings_logit = mask_dict.get('building', torch.zeros(shape, dtype=torch.float32, device=device))
#             grass_logit = mask_dict.get('grass', torch.zeros(shape, dtype=torch.float32, device=device))
#             trail_logit = mask_dict.get('trail or footway', torch.zeros(shape, dtype=torch.float32, device=device))
#             water_logit = mask_dict.get('water', torch.zeros(shape, dtype=torch.float32, device=device))
#             baseball_field_logit = mask_dict.get('baseball field', torch.zeros(shape, dtype=torch.float32, device=device))
#             pond_logit = mask_dict.get('pond', torch.zeros(shape, dtype=torch.float32, device=device))

#             t_l = t_dict.get("t_l", 0.4)
#             t_a = t_dict.get("t_a", 0.6)

#             road_logit = torch.from_numpy(road_logit).to(device)
#             trees_logit = torch.from_numpy(trees_logit).to(device)
#             buildings_logit = torch.from_numpy(buildings_logit).to(device)
#             grass_logit = torch.from_numpy(grass_logit).to(device)
#             trail_logit = torch.from_numpy(trail_logit).to(device)
#             water_logit = torch.from_numpy(water_logit).to(device)
#             baseball_field_logit = torch.from_numpy(baseball_field_logit).to(device)
#             pond_logit = torch.from_numpy(pond_logit).to(device)

#             road_mask = road_logit > t_l
#             trail_mask = trail_logit > t_l
#             grass_mask = grass_logit > t_a
#             buildings_mask = buildings_logit > t_a
#             trees_mask = trees_logit > t_a
#             water_mask = water_logit > t_a
#             baseball_field_mask = baseball_field_logit > t_a
#             pond_mask = pond_logit > t_a




#             # Hierarchy
#             grass_mask = mask_remove(grass_mask, baseball_field_mask)
#             water_mask = mask_remove(water_mask, pond_mask)
#             # Geometry
#             # Nothing for now

#             # Unknown Mask
#             mask_count = road_mask.to(torch.float32) + trail_mask.to(torch.float32) + grass_mask.to(torch.float32) + \
#                         buildings_mask.to(torch.float32) + trees_mask.to(torch.float32) + water_mask.to(torch.float32) \
#                         + baseball_field_mask.to(torch.float32) + pond_mask.to(torch.float32)

#             data_region = (mask_count > 0)
#             data_region_float = data_region.to(torch.float32)


#             costmap = torch.zeros(shape, dtype=torch.float32, device=device)
#             costmap[road_mask] += 0 * road_logit[road_mask]
#             costmap[trail_mask] += 0 * trail_logit[trail_mask]
#             costmap[grass_mask] += 300 * grass_logit[grass_mask]
#             costmap[pond_mask] += 500 * pond_logit[pond_mask]
#             costmap[buildings_mask] += 2000 * buildings_logit[buildings_mask]
#             costmap[trees_mask] += 2000 * trees_logit[trees_mask]
#             costmap[water_mask] += 2000 * water_logit[water_mask]
#             costmap[baseball_field_mask] += 2000 * baseball_field_logit[baseball_field_mask]

#             costmap[data_region] = costmap[data_region] / mask_count[data_region]

#             costmap += costmap.max() * (1 - data_region_float)  # Assign high cost to non-data regions
#             costmap = costmap.cpu().numpy()
#             return costmap
#         </CODE>



#             ACTUAL USER PREFERENCE :

#             "{user_prompt}"

#         default classes : "road", "trail or footway", "water", "grass", "building", and "tree".

#         the classes specified by the user in the actual user preference and use classes in default classes which are not specfied.
#         if default classes are not specified, use the default classes and their values as mentioned above. do not try to define costmap values for classes for which a logit/mask does not exist.

#         for task 1 output should be between the <DICT> and </DICT> markers.
#         for task 2 infer the hierarchies and list them in <HIER> and </HIER> markers. Use these in task 3.
#         for task 3 output should be between the <CODE> and </CODE> markers, output should be python file only with the function and the imports,
#         for task 4 explain the costfunction and tell me whether the generated cost function is correct or not. explain if the cost function actually looked into heirrchy and correctly treat it also make sure you used the default classes and their values as mentioned above.
#         no explanations, no quotes, no extra text.


#         """.replace("{user_prompt}", user_prompt)
#     wrapped = wrap_prompt(prompt)
#     payload = {"texts": [wrapped]}
#     response = requests.post(SERVER_URL, json=payload)
#     response.raise_for_status()
#     data = response.json()
#     print(f"{data}")
#     raw_output = data[0]["outputs"][0]["text"]
#     print(raw_output)

#     class_dict = extract_dict_and_write_code(raw_output, code_filepath)
#     return class_dict


def overseec_query_llm(user_prompt, code_filepath):

    # Use textwrap.dedent to remove the common leading whitespace
    prompt = textwrap.dedent(f"""
        example prefernce : "avoid the baseball field. the pond is dry, so you can go over it. stay on the sides of the road, dont go over water. "

        you have 3 tasks:

        TASK 1 : class segregation
            Thresholds:
            TL = 0.4,
            TA = 0.8

            From this prompt, extract geographic classes, semantic classes and entities (e.g., road, grass, tree, building, water, trails, bushes, river, etc.).
            It can be an areal feature, a linear feature, an object, or a subset of a larger class.

            Use descriptive names if adjectives are provided (e.g., "big trees", "curved roads").

            Do not include specific regions like "center of x", "side of y".


            Always include the default classes:
            "road", "trail or footway", "water", "grass", "building", and "tree".

            For every class, determine if it is:
            - ‘linear/network-like’ (e.g., roads, trails): assign TL
            - ‘areal/blob-like’ (e.g., grass, forest, buildings): assign TA

            Output only the dictionary mapping class names to a list of assigned [threshold, <RGB color>], strictly between the <DICT></DICT> markers.

            example "remain on the outskirts of the forest" [we did not add `outskirt of forest` as it is a specific region, but we added `forest`]

            <DICT>{{
                "road": [0.4, [128, 128, 128]],
                "trail or footway": [0.4, [160, 160, 160]],
                "water": [0.8, [0, 0, 255]],
                "grass": [0.8, [0, 255, 0]],
                "building": [0.8, [255, 0, 0]],
                "tree": [0.8, [34, 139, 34]],
                "baseball field": [0.8, [0, 100, 0]],
                "pond": [0.8, [0, 100, 255]],
                "farm": [0.8, [255, 255, 0]],

            }}</DICT>

        TASK 2 :
        Write down all the heirarchical relationships in the form of a list of tuples. Each tuple should be (parent, child).
        parent is a broad class, child is a more specific instance. Make sure you use only these heirarchy in the code part.
        <HEIR>
            [('water', 'pond'),
             ('grass', 'baseball field')]
        </HEIR>

        TASK 3 :

        Inputs
        - mask_dict: dict mapping lowercase class names → equally sized binary masks (NumPy np.uint8 or PyTorch bool/uint8/float on any device).
        - t_dict: dict with thresholds {{ "t_l": TL, "t_a": TA }} for linear vs areal classes.
        - Helper mask ops are available: mask_and, mask_or, mask_not, and mask_remove(A, B) (i.e., A ∧ ¬B).
        - You may use OpenCV (cv2) for morphology/DT or pure NumPy/PyTorch; autodetect types and stay in one backend.

        Default semantics (when prompt is silent)
        - Costs before normalization:
        road=0, trail or footway=0, grass=100, tree=1000, building=1000, water=1000.

                    Analyze the user's prompt below to understand their preferences. Create a "Costing Plan" that ranks all geographic classes into one of the following tiers. Explain your reasoning.

        * **Tier 1: Preferred (Cost 0-200)**: The most desired terrain. Use for "stay on", "prefer", etc.
        * **Tier 2: Tolerated (Cost 200-50)**: Acceptable, but not ideal. Use for "you can go over", "is fine", etc.
        * **Tier 3: Discouraged (Cost 500-750)**: Use for phrases like, "...if you have to" or "...if road isn't available."
        * **Tier 4: Lethal (Cost 750-900)**: last resort. Use for "avoid", "don't prefer", etc.
        * **Tier 4: Lethal (Cost 900-1000)**: Must be avoided. Use for "don't go on", etc.

        Goal
        Synthesize an executable function f_LLM that translates the user’s NL prompt into a normalized global costmap C ∈ [0, 1000]^(H×W) by following steps (i)–(vi) from the paper.

        ---

        Procedure

        (i) Function signature
        def generate_costmap(mask_dict, t_dict):
            # returns costmap with shape H×W, dtype float32, range [0, 1000]

        (ii) Mask operators
        Use AND, OR, NOT, REMOVE for pixelwise transforms. Implement/consume:
        - mask_and(A,B), mask_or(A,B), mask_not(A), mask_remove(A,B) = A ∧ ¬B.

        (iii) Prompt analysis → weights, hierarchies, geometry
        - Produce per-class weights w_c ∈ [0,1] reflecting preference (lower = more preferred).
        - Map hints like “stay on / prefer / can go over / avoid / don’t go on” into a weight scale (e.g., 0.0, 0.25, 0.5, 0.85, 1.0).
        - Infer semantic hierarchies H: specific classes that are a type of a more general class. parent is a broad class, child is a more specific instance. Use hierarchy from Task 2.
        - Infer geometric cues γ_c (e.g., “stay on the sides of the road”) that require distance fields, dilations, or band masks.

        (iv) Mask operations (hierarchy + geometry)
        1. Thresholding (use TL for linear/network-like, TA for areal/blob-like):
        road, trail -> TL
        grass, tree, building, water, other areal -> TA
        2. Hierarchy enforcement: remove child masks from parents to prevent double counting:
        parent_mask = mask_remove(parent_mask, child_mask) for all (parent, child) ∈ H.
        3. Geometry transforms: for each cue in γ_c, derive auxiliary masks (e.g., distance-to-centerline, bands near road edges via distanceTransform or morphological ops) and incorporate them when computing the class contribution (see v).

        (v) Cost accumulation (per-class contributions, then sum)
        For each class c with probability/logit map P_c (after thresholding → binary mask M_c):
        - Convert any logits to probabilities if needed (sigmoid on float logits; if already binary, cast to float).
        - Compute per-class cost contribution:
        C_c = α_c * w_c * M_c * P_c * G_c
        where:
        - α_c is a base class cost from defaults above (adapted by the prompt when specified).
        - G_c is an optional geometric factor in [0,1] (e.g., band near road gets lower factor if “prefer sides”; centerline gets higher factor if “avoid center”).
        - Sum all C_c pixelwise to get C̃.
        - Unknown region handling: pixels with no class evidence receive a high fallback (e.g., max(C̃)) to discourage exploration outside known data.

        (vi) Normalization
        - Normalize C̃ to [0,1] within the image (robust min–max; guard against all-zeros).
        - Scale to [0,1000] → final C. Ensure dtype float32.

        ---

        def generate_costmap(mask_dict, t_dict):
            # mask operations

            return costmap

        example of a costmap is :
        example prefernce : "dont go over the baseball field. the pond is dry, so you can go over it."

        Example implementation

<CODE>
import numpy as np
import cv2
import torch

def mask_and(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    return torch.logical_and(mask1.bool(), mask2.bool()).to(torch.uint8)

def mask_or(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(mask1.bool(), mask2.bool()).to(torch.uint8)

def mask_not(mask: torch.Tensor) -> torch.Tensor:
    return torch.logical_not(mask.bool()).to(torch.uint8)

def mask_remove(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    return (mask1.bool() & ~mask2.bool()).to(torch.uint8)

def convert_masks2torch(mask_dict, device):
    for key in mask_dict:
        if isinstance(mask_dict[key], np.ndarray):
            mask_dict[key] = torch.from_numpy(mask_dict[key]).to(device)
    return mask_dict


def generate_costmap(mask_dict, t_dict={{"t_l":0.4, "t_a":0.6}}, device="cpu"):
    shape = next(iter(mask_dict.values())).shape
    mask_dict = convert_masks2torch(mask_dict, device)

    road_logit = mask_dict.get('road', torch.zeros(shape, dtype=torch.float32, device=device))
    trees_logit = mask_dict.get('tree', torch.zeros(shape, dtype=torch.float32, device=device))
    buildings_logit = mask_dict.get('building', torch.zeros(shape, dtype=torch.float32, device=device))
    grass_logit = mask_dict.get('grass', torch.zeros(shape, dtype=torch.float32, device=device))
    trail_logit = mask_dict.get('trail or footway', torch.zeros(shape, dtype=torch.float32, device=device))
    water_logit = mask_dict.get('water', torch.zeros(shape, dtype=torch.float32, device=device))
    baseball_field_logit = mask_dict.get('baseball field', torch.zeros(shape, dtype=torch.float32, device=device))
    pond_logit = mask_dict.get('pond', torch.zeros(shape, dtype=torch.float32, device=device))

    t_l = t_dict.get("t_l", 0.4)
    t_a = t_dict.get("t_a", 0.6)

    road_mask = road_logit > t_l
    trail_mask = trail_logit > t_l
    grass_mask = grass_logit > t_a
    buildings_mask = buildings_logit > t_a
    trees_mask = trees_logit > t_a
    water_mask = water_logit > t_a
    baseball_field_mask = baseball_field_logit > t_a
    pond_mask = pond_logit > t_a




    # Hierarchy
    grass_mask = mask_remove(grass_mask, baseball_field_mask)
    water_mask = mask_remove(water_mask, pond_mask)
    # Geometry
    # Nothing for now

    # Unknown Mask
    mask_count = road_mask.to(torch.float32) + trail_mask.to(torch.float32) + grass_mask.to(torch.float32) + \
                buildings_mask.to(torch.float32) + trees_mask.to(torch.float32) + water_mask.to(torch.float32) \
                + baseball_field_mask.to(torch.float32) + pond_mask.to(torch.float32)

    data_region = (mask_count > 0)
    data_region_float = data_region.to(torch.float32)


    costmap = torch.zeros(shape, dtype=torch.float32, device=device)
    costmap[road_mask] += 0 * road_logit[road_mask]
    costmap[trail_mask] += 0 * trail_logit[trail_mask]
    costmap[grass_mask] += 300 * grass_logit[grass_mask]
    costmap[pond_mask] += 500 * pond_logit[pond_mask]
    costmap[buildings_mask] += 2000 * buildings_logit[buildings_mask]
    costmap[trees_mask] += 2000 * trees_logit[trees_mask]
    costmap[water_mask] += 2000 * water_logit[water_mask]
    costmap[baseball_field_mask] += 2000 * baseball_field_logit[baseball_field_mask]

    costmap[data_region] = costmap[data_region] / mask_count[data_region]

    costmap += costmap.max() * (1 - data_region_float)  # Assign high cost to non-data regions
    costmap = costmap.cpu().numpy()
    return costmap
</CODE>



        <USER_PROMPT>

        "{user_prompt}"

        </USER_PROMPT>

        default classes : "road", "trail or footway", "water", "grass", "building", and "tree".
        use the classes only specified by the user between the tags <USER_PROMPT> </USER_PROMPT>.
        if default classes are not specified, use the default classes and their values as mentioned above.

        PLEASE USE THE DEFAULT CLASSES AND THEIR VALUES IF THEY ARE NOT SPECIFIED IN THE USER PROMPT.
        for task 1 output should be between the <DICT> and </DICT> markers. Recheck if any classes it outside the classes specifed in the <USER_PROMPT> </USER_PROMPT> and default classes.
        for task 2 infer the hierarchies and list them in <HIER> and </HIER> markers. Use these in task 3.
        for task 3 output should be between the <CODE> and </CODE> markers, output should be python file only with the function and the imports,
        for task 4 explain the costfunction and tell me whether the generated cost function is correct or not. explain if the cost function actually looked into heirrchy and correctly treat it also make sure you used the default classes and their values as mentioned above.
        no explanations, no quotes, no extra text.

        """)

    # I also had to fix your f-string by doubling the curly braces {{ }} inside the prompt
    # so they are treated as literal characters, not placeholders.

    wrapped = wrap_prompt(prompt)
    payload = {"texts": [wrapped]}
    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    # print(f"{data}")
    raw_output = data[0]["outputs"][0]["text"]
    print(raw_output)

    class_dict = extract_dict_and_write_code(raw_output, code_filepath)
    return class_dict

if __name__ == "__main__":
    # Example list of prompts
    # prompt = "I prefer the road try to walk on pavements as possible, but road is fine as well. Grass is okay i guess, but please try to avoid the trees. Also gravel can be bit difficult for us"
    # prompt = "I prefer the trees"
    # prompt = "I prefer the roads, grass is okay, but please avoid the baseball field"
    # prompt = "go over the river and water is lethal"
    prompt = "Stay on the road. No Grass."
    # prompt = "avoid the river"

    # pref_score_dict = class_segregation_prompt(prompt)

    overseec_query_llm(prompt, "generated_costmap.py")
