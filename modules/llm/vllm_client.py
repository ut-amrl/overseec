import ast
import requests

SERVER_URL = "http://127.0.0.1:8000/generate"  # Update if running on a different host

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

def overseec_query_llm(user_prompt, code_filepath):

    prompt = """

        example prefernce : "avoid the baseball field. the pond is dry, so you can go over it. stay on the sides of the road, dont go over water. "

        you have 2 tasks:

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

            <DICT>{
                "road": [0.4, [128, 128, 128]],
                "trail or footway": [0.4, [160, 160, 160]],
                "water": [0.8, [0, 0, 255]],
                "grass": [0.8, [0, 255, 0]],
                "building": [0.8, [255, 0, 0]],
                "tree": [0.8, [34, 139, 34]],
                "baseball field": [0.8, [0, 100, 0]],
                "pond": [0.8, [0, 100, 255]],
                "farm": [0.8, [255, 255, 0]],

            }</DICT>

        
        TASK 2 : generate costmap

            Inputs
            • mask_dict: a dict whose keys are lowercase class names (e.g., "road", "tree") and whose values are same-sized binary NumPy arrays.
            • Three Boolean helpers are already defined: mask_and, mask_or, mask_not.
            • You may use OpenCV for dilation, blurring, or distance transforms.

            Default class meanings (apply only when the user says nothing about that class):
            You should change these values according to user specification otherwise use these values.
            
            default `road` : cost 0
            default `trail` or footway : cost 0 
            default `grass` : cost 100
            default `tree` : cost 1000
            default `building` : cost 1000
            default `water` : cost 1000


            Analyze the user's prompt below to understand their preferences. Create a "Costing Plan" that ranks all geographic classes into one of the following tiers. Explain your reasoning.

            * **Tier 1: Preferred (Cost 0-200)**: The most desired terrain. Use for "stay on", "prefer", etc.
            * **Tier 2: Tolerated (Cost 200-50)**: Acceptable, but not ideal. Use for "you can go over", "is fine", etc.
            * **Tier 3: Discouraged (Cost 500-750)**: Use for phrases like, "...if you have to" or "...if road isn't available."
            * **Tier 4: Lethal (Cost 750-900)**: last resort. Use for "avoid", "don't prefer", etc.
            * **Tier 4: Lethal (Cost 900-1000)**: Must be avoided. Use for "don't go on", etc.

            ---------------------------------------

            ** Function signature **

            def generate_costmap(mask_dict):

                # mask operations

                return costmap

            example of a costmap is :

            example prefernce : "dont go over the baseball field. the pond is dry, so you can go over it."


            <CODE>
            import numpy as np
            import cv2

            def mask_and(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
                return np.logical_and(mask1, mask2).astype(np.uint8)

            def mask_or(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
                return np.logical_or(mask1, mask2).astype(np.uint8)

            def mask_not(mask: np.ndarray) -> np.ndarray:
                return np.logical_not(mask).astype(np.uint8)
            
            def remove_mask(mask1 : np.ndarray, mask2: np.ndarray) -> np.ndarray:
                return np.logical_and(mask1, mask_not(mask2)).astype(np.uint8)
            
            def distance_from_center_blob_mask(mask: np.ndarray) -> np.ndarray:
                # Ensure mask is binary uint8
                mask = mask.astype(np.uint8)

                # Compute distance transform inside the blob (non-zero regions)
                mask_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

                # Normalize distance to [0, 1]
                max_dist = mask_dist.max() or 1.0
                normalized_dist = mask_dist / max_dist

                return normalized_dist

            def generate_costmap(mask_dict):

                shape = next(iter(mask_dict.values())).shape

                classes = ['road', 'trail or footway', 'water', 'grass', 'building', 'pond', 'baseball field']
                assert all(cls in mask_dict for cls in classes), "mask_dict must contain all default classes"

                road_mask = mask_dict.get('road', np.zeros(shape, dtype=np.float32))
                trees_mask = mask_dict.get('tree', np.zeros(shape, dtype=np.float32))
                buildings_mask = mask_dict.get('building', np.zeros(shape, dtype=np.float32))
                grass_mask = mask_dict.get('grass', np.zeros(shape, dtype=np.float32))
                trail_mask = mask_dict.get('trail or footway', np.zeros(shape, dtype=np.float32))
                water_mask = mask_dict.get('water', np.zeros(shape, dtype=np.float32))
                pond_mask = mask_dict.get('river', np.zeros(shape, dtype=np.float32))
                baseball_mask = mask_dict.get('baseball field', np.zeros(shape, dtype=np.float32))

                postive_hierarchy = [('baseball field', 'grass')]
                negative_hierarchy = [('pond', 'water')]

                # rectifying water mask
                water_mask = remove_mask(water_mask, pond_mask)

                # rectifying grass
                grass_mask = remove_mask(grass_mask, baseball_mask)

                # Step 1: Define non-avoid mask

                lethal_mask = mask_or(mask_or(mask_or(trees_mask, buildings_mask), water_mask).
                non_lethal_mask = mask_not(lethal_mask)
                lethal_mask = lethal_mask.astype(np.float32)
                non_lethal_mask = non_lethal_mask.astype(np.float32)
                

                costmap = np.ones(shape, dtype=np.float32) * 1000.0
                costmap[road_mask.astype(bool)] = 0.0

                center_of_road_mask = distance_from_center_blob_mask(road_mask)
                costmap += 50 * center_of_road_mask.astype(np.float32)  # prefer the sides of the road
                # costmap += 50 *(1 - center_of_road_mask).astype(np.float32)  # prefer the center of the road

                
                costmap[trail_mask.astype(bool)] = 0.0
                costmap[grass_mask.astype(bool)] = 100.0

                costmap[baseball_mask.astype(bool)] = 700.0  # baseball field

                costmap[pond_mask.astype(bool)] = 0.0  # pond
                
                costmap[lethal_mask.astype(bool)] = 1000.0  # Avoidance cost for trees, buildings, and water
                                

                return costmap

            </CODE>

            ACTUAL USER PREFERENCE :

            "{user_prompt}"

        default classes : "road", "trail or footway", "water", "grass", "building", and "tree".

        the classes specified by the user int the actual user preference and use classes in default classes which are not specfied.
        if default classes are not specified, use the default classes and their values as mentioned above.

        for task 1 output should be between the <DICT> and </DICT> markers.
        for task 2 output should be between the <CODE> and </CODE> markers, output should be python file only with the function and the imports,
        for task 3 explain the costfunction and tell me whether the generated cost function is correct or not. explain if the cost function actually looked into heirrchy and correctly treat it also make sure you used the default classes and their values as mentioned above.
        no explanations, no quotes, no extra text.
        
        
        """.replace("{user_prompt}", user_prompt)
    wrapped = wrap_prompt(prompt)
    payload = {"texts": [wrapped]}
    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()
    data = response.json()
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
    prompt = "avoid the water but the river has dried up so you can go over it"
    # prompt = "avoid the river"

    # pref_score_dict = class_segregation_prompt(prompt)

    overseec_query_llm(prompt, "generated_costmap.py")