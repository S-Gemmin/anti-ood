### FORTRESS: Anticipatory OOD Safety

#### Experiments

##### Synthetic (2D Gaussian)
```bash
python main.py --synthetic
```

##### Real Embeddings (Text)
```bash
python main.py --real
```

##### Real Embeddings (Image)
```bash
python image_embedding_experiment.py --embeddings safe_embeddings.npy --vel 0.001 --beta 1.5
```

##### Unity

First, all the relevant files are in the Unity folder. To get started, download [Unity Hub](https://unity.com/download) and install Unity 6 (or 2023 LTS), then clone the [ML-Agents repository](https://github.com/Unity-Technologies/ml-agents) and follow its package installation guide to add `com.unity.ml-agents` to your project via the Package Manager. Open the ML-Agents **Kart Racing** template scene (found under `Assets/ML-Agents/Examples/Kart`), then add `SunsetController.cs` to your Directional Light object, add `BrightnessMonitor.cs`, `SafeFallback.cs`, `FORTRESSController.cs`, `ProactiveRiskController.cs`, and `OODExperimentLogger.cs` to the `KartAgent` prefab, assign the scene's start point transform to the `Safe Target` field on `SafeFallback`, and assign the agent's camera to the `Target Camera` field on `BrightnessMonitor`, then hit Play and the sunset will begin automatically, triggering whichever controller you have enabled to return the kart to safety.

In addition, the light proxy validation notebook is called `brightness_ood.ipynb`, which you can simply run after installing the required dependencies. 

###### Demo
![Trajectory Visualization](results/unity.gif)
