"""Perception Module for Multimodal Embodiment (Sight, Sound & Screen)."""

from .vision_encoder import VisionPatchTransformer
from .continuous_visual_eye import (
    ContinuousVisualEye,
    ScreenCaptureEngine,
    VisualDesignLearner,
    DesignKnowledgeStore,
    ScreenFrame,
    LearnedDesignArchetype,
    ObservedVisualContext,
)
from .dual_stream_ingestor import (
    DualStreamCognitiveIngestor,
    DualStreamArticleDocument,
    DualStreamIngestionResult,
    ArticleSectionNode,
    MathematicalBlock,
    VisualDiagramNode,
)
from .neural_vocal_tract import (
    NeuralVocalTract,
    VocalCadenceProfile,
    LiljencrantsFantGlottalSource,
    VocalCavityAcousticFilter,
    NeuralManifoldArticulator,
    TurbulentAspirationGenerator,
    FormantTarget,
    STANDARD_FORMANT_TARGETS,
    AdaptiveEchoCanceller,
    FullDuplexAcousticStream,
)
from .spatial_3d_hologram import (
    Spatial3DHolographicEngine,
    PointCloud3DEncoder,
    Deictic3DGestureTracker,
    Volumetric3DKinematics,
    Spatial3DObject,
    Raycast3DHit,
)

__all__ = [
    "VisionPatchTransformer",
    "ContinuousVisualEye",
    "ScreenCaptureEngine",
    "VisualDesignLearner",
    "DesignKnowledgeStore",
    "ScreenFrame",
    "LearnedDesignArchetype",
    "ObservedVisualContext",
    "DualStreamCognitiveIngestor",
    "DualStreamArticleDocument",
    "DualStreamIngestionResult",
    "ArticleSectionNode",
    "MathematicalBlock",
    "VisualDiagramNode",
    "NeuralVocalTract",
    "VocalCadenceProfile",
    "LiljencrantsFantGlottalSource",
    "VocalCavityAcousticFilter",
    "NeuralManifoldArticulator",
    "TurbulentAspirationGenerator",
    "FormantTarget",
    "STANDARD_FORMANT_TARGETS",
    "AdaptiveEchoCanceller",
    "FullDuplexAcousticStream",
    "Spatial3DHolographicEngine",
    "PointCloud3DEncoder",
    "Deictic3DGestureTracker",
    "Volumetric3DKinematics",
    "Spatial3DObject",
    "Raycast3DHit",
]


