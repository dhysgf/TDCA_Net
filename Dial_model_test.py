import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import logging
import random
from dataclasses import dataclass, field
from models.base import generate_cca_references
from models.tdca import TDCA
from data_loader.Dial_laod import getSSVEP12Intra
from network.CNN_model import TENet, LinearWithConstraint
@dataclass
class SSVEPConfig:
    dataset: str = 'Benchmark'
    epochs1: int = 500
    epochs2: int = 500
    bz1: int = 64
    bz2: int = 12
    lr: float = 0.001
    trans_lr: float = 0.0001
    Fs: int = 256  # 注意：这里是大写 F
    Nc: int = 8
    Nf: int = 12
    filters:int = 32
    Ns: int = 10
    wd: float = 0.0003
    data_root: str = "../data"
    window_times: List[float] = None
    data_lengths: List[int] = field(init=False)
    freq_list: List[float] = None
    def __post_init__(self):
        if self.freq_list is None:
            self.freq_list = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                              10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
        if self.window_times is None:
            self.window_times = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.data_lengths = []
        K1, K2, S, filters = 7, 3, 3, 32
        for wt in self.window_times:
            Nt = int(wt * self.Fs) * 2
            after_conv1 = (Nt - K1) // S + 1
            after_conv2 = (after_conv1 - K2) // S + 1
            self.data_lengths.append(filters * after_conv2)


# ==================== 数据管理器 ====================
class SSVEPDataManager:
    """数据加载、TDCA特征提取和预处理"""

    def __init__(self, config: SSVEPConfig):
        self.config = config
        self.data_root = Path(config.data_root).resolve()
        self.logger = logging.getLogger(__name__)

    def load_subject_data(self, subject_id: int, train_ratio: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        dataset = getSSVEP12Intra(
            subject=subject_id,
            train_ratio=train_ratio,
            mode=mode,
            data_root=str(self.data_root)
        )
        return dataset.eeg_data.numpy(), dataset.label_data.numpy()

    def extract_tdca_features(self, train_data: np.ndarray, train_labels: np.ndarray,
                              test_data: np.ndarray, Yf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        estimator = TDCA(padding_len=5, n_components=8)
        estimator.fit(X=train_data, y=train_labels, Yf=Yf)
        _, X_train_trans = estimator.transform(train_data)
        _, X_test_trans = estimator.transform(test_data)
        X_train_trans = np.squeeze(X_train_trans, axis=2)
        X_test_trans = np.squeeze(X_test_trans, axis=2)
        return X_train_trans, X_test_trans

    def prepare_cross_subject_data(self, test_subject: int, window_time: float
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        train_data, train_labels = [], []
        for subj in range(1, self.config.Ns + 1):
            if subj != test_subject:
                data, labels = self.load_subject_data(subj, train_ratio=1.0, mode='train')
                data = self._slice_window(data, window_time)
                train_data.append(data)
                train_labels.append(labels)
        X_train = np.concatenate(train_data, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
        y_train = np.squeeze(y_train, axis=1)
        return X_train, y_train

    def prepare_intra_subject_data(self, test_subject: int, train_ratio: float, window_time: float
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_data, train_labels = self.load_subject_data(test_subject, train_ratio=train_ratio, mode='train')
        test_data, test_labels = self.load_subject_data(test_subject, train_ratio=train_ratio, mode='test')
        train_data = self._slice_window(train_data, window_time)
        test_data = self._slice_window(test_data, window_time)
        train_labels = np.squeeze(np.array(train_labels), axis=1)
        test_labels = np.squeeze(np.array(test_labels), axis=1)
        return train_data, train_labels, test_data, test_labels

    def _slice_window(self, data: np.ndarray, window_time: float) -> np.ndarray:
        start_idx = int(0.135 * self.config.Fs)  # 使用大写的 Fs
        end_idx = start_idx + int(window_time * self.config.Fs) + 5
        return np.array(data[:, :, start_idx:end_idx])

class SSVEPModelManager:
    def __init__(self, config: SSVEPConfig, model_path: str = "../temp"):
        self.config = config
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def create_model(self) -> torch.nn.Module:
        model = TENet(num_classes=self.config.Nf, num_channels=self.config.Nc)
        return model

    def save_model(self, model: torch.nn.Module, stage: str):
        save_path = self.model_path / f"net_{stage}.pth"
        torch.save(model.state_dict(), save_path)
        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, stage: str, data_length: int) -> torch.nn.Module:
        model = self.create_model()
        load_path = self.model_path / f"net_{stage}.pth"
        model.load_state_dict(torch.load(load_path))
        self.logger.info(f"Model loaded from {load_path}")
        return model

class SSVEPTrainer:
    def __init__(self, config: SSVEPConfig, device: str):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)

    def create_dataloader(self, data: np.ndarray, labels: np.ndarray,
                          batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(data, labels)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=0,
            worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id)
        )

    def train_stage(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                    valid_loader: torch.utils.data.DataLoader, epochs: int, lr: float
                    ) -> Tuple[float, float, float, float]:
        from Train.Trans_Trainer import train_on_batch
        return train_on_batch(epochs, train_loader, valid_loader, lr, model, self.device)

    def train_two_stage(self, X_train: np.ndarray, y_train: np.ndarray,
                        train_data: np.ndarray, train_labels: np.ndarray,
                        test_data: np.ndarray, test_labels: np.ndarray,
                        Yf: np.ndarray, win_len: int
                        ) -> Tuple[float, float, float, float]:
        """
        两阶段训练
        """
        # 阶段1: Cross-Subject Training
        self.logger.info("Stage 1: Cross-Subject Training")
        # 使用大写 Fs
        print(f"采样率: {self.config.Fs}")  # 修复：使用大写的 Fs
        # Fit TDCA on raw cross-subject data
        estimator1 = TDCA(padding_len=5, n_components=8)
        estimator1.fit(X_train, y_train, Yf)
        _, X_cross_train = estimator1.transform(X_train)
        _, X_intra_test = estimator1.transform(test_data)
        X_cross_train = np.squeeze(X_cross_train, axis=2)
        X_intra_test = np.squeeze(X_intra_test, axis=2)
        model = TENet(num_classes=self.config.Nf, num_channels=self.config.Nc,filters=self.config.filters)
        model = model.to(self.device)
        train_loader = self.create_dataloader(X_cross_train, y_train, self.config.bz1, True)
        test_loader = self.create_dataloader(X_intra_test, test_labels, 24, False)

        val_acc, kappa, recall, precision = self.train_stage(
            model, train_loader, test_loader, self.config.epochs1, self.config.lr
        )
        # 阶段2: Intra-Subject Fine-tuning
        self.logger.info("Stage 2: Intra-Subject Fine-tuning")

        # Fit NEW TDCA on raw intra-subject training data
        estimator2 = TDCA(padding_len=5, n_components=8)
        estimator2.fit(train_data, train_labels, Yf)
        _, X_intra_train = estimator2.transform(train_data)
        _, X_intra_test2 = estimator2.transform(test_data)
        X_intra_train = np.squeeze(X_intra_train, axis=2)
        X_intra_test2 = np.squeeze(X_intra_test2, axis=2)

        # 替换全连接层 - 确保使用正确的输入维度
        # 首先进行一次前向传播来获取实际的特征维度
        model.eval()
        with torch.no_grad():
            test_input = torch.tensor(X_intra_train[:1], dtype=torch.float32).to(self.device)
            # 确保输入形状正确 [batch, Nm, Nc, Nt]
            if test_input.dim() == 3:
                test_input = test_input.unsqueeze(1)  # 添加 Nm 维度
            features = model.chan_feature_extraction(test_input)
            features = model.time1_feature_extraction(features)
            features = model.time2_feature_extraction(features)
            actual_feature_dim = features.view(features.size(0), -1).shape[1]
        # 使用实际计算的特征维度替换全连接层
        model.dense_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            LinearWithConstraint(actual_feature_dim, self.config.Nf, max_norm=1.0)
        ).to(self.device)
        model = model.to(self.device)
        train_loader = self.create_dataloader(X_intra_train, train_labels, self.config.bz2, True)
        test_loader = self.create_dataloader(X_intra_test2, test_labels, 24, False)

        val_acc, kappa, recall, precision = self.train_stage(
            model, train_loader, test_loader, self.config.epochs2, self.config.trans_lr
        )
        return val_acc, kappa, recall, precision

class SSVEPExperiment:
    def __init__(self, config: SSVEPConfig, output_dir: str = "../results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_manager = SSVEPDataManager(config)
        self.model_manager = SSVEPModelManager(config)
        self.trainer = SSVEPTrainer(config, self.device)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self._set_random_seeds(42)
    def _set_random_seeds(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def generate_reference_signals(self, window_time: float) -> np.ndarray:
        return generate_cca_references(
            freqs=self.config.freq_list, srate=256, T=window_time, n_harmonics=5
        )
    def evaluate_performance(self, acc: float, window_time: float) -> float:
        from Utils.itr import itr
        return itr(self.config.Nf, float(acc), 0.5 + window_time)
    def save_results(self, results: dict, window_time: float):
        for metric, values in results.items():
            df = pd.DataFrame(values)
            save_path = self.output_dir / f"{window_time:.1f}s_{metric}.csv"
            df.to_csv(save_path, index=False)
            self.logger.info(f"Saved {metric} to {save_path}")
    def run(self):
        train_ratio = 0.5#训练集占比可以按照实际手动设置
        self.logger.info("Starting SSVEP Experiment")
        if self.config.window_times is None or self.config.data_lengths is None:
            raise ValueError("window_times和data_lengths未正确初始化")
        for idx, window_time in enumerate(self.config.window_times):
            self.logger.info(f"\n=== Window Time: {window_time}s (data_length: {self.config.data_lengths[idx]}) ===")
            Yf = self.generate_reference_signals(window_time)
            results = {metric: [] for metric in ['acc', 'itr', 'kappa', 'precision', 'recall']}
            for test_subject in range(1, self.config.Ns + 1):
                self.logger.info(f"Test Subject: {test_subject}")
                X_train, y_train = self.data_manager.prepare_cross_subject_data(
                    test_subject, window_time,
                )
                train_data, train_labels, test_data, test_labels = \
                    self.data_manager.prepare_intra_subject_data(
                        test_subject, train_ratio, window_time,
                    )
                val_acc, kappa, recall, precision = self.trainer.train_two_stage(
                    X_train, y_train, train_data, train_labels,
                    test_data, test_labels, Yf, self.config.data_lengths[idx]
                )
                itr_val = self.evaluate_performance(val_acc, window_time)
                results['acc'].append(val_acc)
                results['itr'].append(itr_val)
                results['kappa'].append(kappa)
                results['precision'].append(precision)
                results['recall'].append(recall)
                self.logger.info(
                    f"Subject {test_subject}: ACC={val_acc:.4f} | ITR={itr_val:.2f} | Kappa={kappa:.4f}"
                )
            self.save_results(results, window_time)
            self.logger.info(
                f"=== Summary for {window_time}s ===\n"
                f"Average ACC: {np.mean(results['acc']):.4f} | "
                f"Average ITR: {np.mean(results['itr']):.2f} | "
                f"Average Kappa: {np.mean(results['kappa']):.4f}"
            )
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_root = str(current_dir / "data")
    config = SSVEPConfig(
        Fs=256, Nc=8, Nf=12, Ns=10, epochs1=500, epochs2=500,
        bz1=64, bz2=12, lr=0.001, trans_lr=0.0001,
        data_root=data_root,
        window_times=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    print(f"当前工作目录: {current_dir.absolute()}")
    print(f"数据根目录: {config.data_root}")
    print(f"绝对路径: {Path(config.data_root).absolute()}")
    print(f"窗口时间: {config.window_times}")
    print(f"自动计算长度: {config.data_lengths}")
    experiment = SSVEPExperiment(config, output_dir="out_files_path")#可以根据实际路径设置文件保存的路径
    experiment.run()