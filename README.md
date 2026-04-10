### FORTRESS: Anticipatory OOD Safety

## Experiments

### Synthetic (2D Gaussian)
```bash
python main.py --synthetic
```

### Real Embeddings (Text)
```bash
python main.py --real
```

### Real Embeddings (Image)
```bash
python image_embedding_experiment.py --embeddings safe_embeddings.npy --vel 0.001 --beta 1.5
```
## Demo
![Trajectory Visualization](results/unity.gif)
