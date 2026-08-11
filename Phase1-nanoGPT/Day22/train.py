import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    print(f"learning_rate: {cfg.train.learning_rate}")
    print(f"batch_size: {cfg.train.batch_size}")
    print(f"n_layer: {cfg.model.n_layer}")

    return cfg

if __name__ == "__main__":
    cfg = train() 
    