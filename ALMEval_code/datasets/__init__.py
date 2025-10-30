from .starbench import (
    TemporalReasoningDataset,
    TemporalReasoningGivenCaptionDataset,
    TemporalReasoningGivenUncutAudioDataset,
    SpatialReasoningDataset,
    SpatialReasoningChannelwiseDataset,
    PerceptionDataset,
    PerceptionSpatialDataset,
    PerceptionNonSpatialDataset
)

# Central registry mapping short aliases to Dataset classes
DATASET_REGISTRY = {
    # Temporal Reasoning
    'tr': TemporalReasoningDataset,
    'tr_cap': TemporalReasoningGivenCaptionDataset,
    'tr_uncut': TemporalReasoningGivenUncutAudioDataset,
    # Spatial Reasoning
    'sr': SpatialReasoningDataset,
    'sr_ch': SpatialReasoningChannelwiseDataset,
    # Perception
    'pc': PerceptionDataset,
    'pc_sp': PerceptionSpatialDataset,
    'pc_nsp': PerceptionNonSpatialDataset,
}

# Groups of aliases for convenience
DATASET_GROUPS = {
    "temporal_all": ['tr', 'tr_cap', 'tr_uncut'],
    "spatial_all": ['sp', 'sp_ch'],
    "perception_all": ['pc_sp', 'pc_nsp'],
    "starbench_all": [
        'tr', 'tr_cap', 'tr_uncut',
        'sr', 'sr_ch',
        'pc_sp', 'pc_nsp'
    ],
    "starbench_default": ['pc', 'tr', 'sr']
}

def build_dataset(dataset_names: list, dataset_root: str) -> list:
    """
    Builds a list of dataset objects from a list of aliases or group names.
    
    Args:
        dataset_names (list): A list of strings, e.g., ['tr', 'sp_ch', 'perception_all'].
        dataset_root (str): The root directory for datasets.

    Returns:
        list: A list of instantiated dataset objects.
    """
    final_aliases = []
    seen= set()
    for name in dataset_names:
        # Handle multiple datasets joined by '+'
        sub_names = [n.strip().lower() for n in name.split('+')]
        for sub_name in sub_names:
            if sub_name in DATASET_GROUPS:
                for x in DATASET_GROUPS[sub_name]:
                    if x not in seen:
                        final_aliases.append(x)
                        seen.add(x)
            elif sub_name in DATASET_REGISTRY and sub_name not in seen:
                final_aliases.append(sub_name)
                seen.add(sub_name)
            else:
                raise ValueError(f"Unknown dataset alias or group: '{sub_name}'")
    
    dataset_objects = []

    for alias in list(final_aliases):
        dataset_class = DATASET_REGISTRY[alias]
        dataset_objects.append(dataset_class(dataset_root=dataset_root))
        
    return dataset_objects