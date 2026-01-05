import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import logging
import random
from dataclasses import dataclass, field
from data_loader.Benchmark_load import Data_load
from models.base import generate_cca_references
from models.tdca import TDCA
from network.CNN_model import TENet, LinearWithConstraint
@dataclass
class SSVEPConfig:
    dataset: str = 'Benchmark'  # 可以改为 'NewDataset'
    epochs1: int = 500
    epochs2: int = 500
    bz1: int = 64
    bz2: int = 12
    lr: float = 0.01
    trans_lr: float = 0.0001
    Fs: int = 250
    Nc: int = 8
    Nf: int = 12
    filters: int = 40
    Ns: int = 10
    wd: float = 0.0003
    data_root: str = "../data"#改成自己的数据路径
    window_times: List[float] = None
    data_lengths: List[int] = field(init=False)
    freq_list: List[float] = None
   # 新数据集的特定参数
    new_dataset_params: dict = field(default_factory=dict)
    def __post_init__(self):
        if self.freq_list is None:
            if self.dataset == 'Benchmark':
                self.freq_list=[8., 9., 10., 11., 12., 13., 14., 15., 8.2, 9.2, 10.2,
             11.2, 12.2, 13.2, 14.2, 15.2, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4,
             14.4, 15.4, 8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6, 8.8,
             9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8]
        if self.window_times is None:
            self.window_times = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.data_lengths = []
        K1, K2, S, filters = 7, 3, 3, 32
        for wt in self.window_times:
            Nt = int(wt * self.Fs) * 2
            after_conv1 = (Nt - K1) // S + 1
            after_conv2 = (after_conv1 - K2) // S + 1
            self.data_lengths.append(filters * after_conv2)
class SSVEPDataManager:
    """数据加载、TDCA特征提取和预处理"""
    def __init__(self, config: SSVEPConfig):
        self.config = config
        self.data_root = Path(config.data_root).resolve()
        self.logger = logging.getLogger(__name__)
    def load_Benchmark_data(self, subject_id: int, train_block: int, test_block: int, mode: str) -> Tuple[
        np.ndarray, np.ndarray]:
        """加载Benchmark数据集"""
        data_path = self.data_root / f"S{subject_id}.mat"
        train_data, train_labels, test_data, test_labels = Data_load(
            str(data_path),
            train_balock=train_block,
            test_block=test_block,
            ws=2.0
        )
        # 检查返回数据的维度
        self.logger.debug(
            f"Benchmark S{subject_id} - Train data: {train_data.shape}, Train labels: {train_labels.shape}")
        self.logger.debug(f"Benchmark S{subject_id} - Test data: {test_data.shape}, Test labels: {test_labels.shape}")
        if mode == 'train':
            return train_data, train_labels
        else:
            return test_data, test_labels

    def load_subject_data(self, subject_id: int, train_ratio: float = None,
                          train_block: int = None, test_block: int = None, mode: str = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """根据配置选择数据加载函数"""
        if self.config.dataset == 'Benchmark':
            if train_block is not None and test_block is not None and mode is not None:
                return self.load_Benchmark_data(subject_id, train_block, test_block, mode)
            else:
                raise ValueError("Benchmark数据集需要 train_block, test_block 和 mode 参数")
        else:
            raise ValueError(f"未知的数据集: {self.config.dataset}")
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
                if self.config.dataset == "Benchmark":
                    data, labels = self.load_subject_data(subj, train_block=6, test_block=0, mode='train')
                    if labels.ndim == 1:
                        labels = labels.reshape(-1, 1)
                    data = self._slice_window(data, window_time)
                    train_data.append(data)
                    train_labels.append(labels)
                else:
                    raise ValueError("未知数据集")
        if not train_data:
            raise ValueError(f"没有收集到任何训练数据（测试被试: {test_subject}）")
        X_train = np.concatenate(train_data, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
        if y_train.ndim > 1 and y_train.shape[1] == 1:
            y_train = np.squeeze(y_train, axis=1)
        elif y_train.ndim == 2 and y_train.shape[1] > 1:
            self.logger.warning(f"标签维度异常: {y_train.shape}，可能存在问题")
        self.logger.debug(f"Cross-subject data shape: {X_train.shape}, labels shape: {y_train.shape}")
        return X_train, y_train
    def prepare_intra_subject_data(self, test_subject: int,
                                   train_block: int = None, test_block: int = None, window_time: float = 0
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if self.config.dataset == "Benchmark":
            if train_block is None or test_block is None:
                raise ValueError("Benchmark数据集需要提供 train_block 和 test_block 参数")
            train_data, train_labels = self.load_subject_data(
                subject_id=test_subject, train_block=train_block, test_block=test_block, mode='train'
            )
            test_data, test_labels = self.load_subject_data(
                subject_id=test_subject, train_block=train_block, test_block=test_block, mode='test'
            )
        else:
            raise ValueError(f"未知的数据集: {self.config.dataset}")
        if train_labels.ndim == 1:
            train_labels = train_labels.reshape(-1, 1)
        if test_labels.ndim == 1:
            test_labels = test_labels.reshape(-1, 1)
        train_data = self._slice_window(train_data, window_time)
        test_data = self._slice_window(test_data, window_time)
        train_labels = np.squeeze(train_labels) if train_labels.shape[1] == 1 else train_labels
        test_labels = np.squeeze(test_labels) if test_labels.shape[1] == 1 else test_labels
        self.logger.debug(f"Intra-subject - Train: {train_data.shape}, Test: {test_data.shape}")
        self.logger.debug(f"Intra-subject - Train labels: {train_labels.shape}, Test labels: {test_labels.shape}")
        return train_data, train_labels, test_data, test_labels
    def _slice_window(self, data: np.ndarray, window_time: float) -> np.ndarray:
        """数据切片，根据数据集调整切片参数"""
        if self.config.dataset == 'Benchmark':
            start_idx = int(0.635 * self.config.Fs)
        else:
            raise ValueError(f"未知数据集: {self.config.dataset}")
        # 修复：确保数据至少是2D的
        if data.ndim < 2:
            raise ValueError(f"数据维度太低: {data.ndim}D，期望至少2D (channels, time)")
        elif data.ndim == 2:
            data = data[np.newaxis, :, :]
        end_idx = start_idx + int(window_time * self.config.Fs)+5

        if end_idx > data.shape[-1]:
            self.logger.warning(f"窗口越界: end_idx={end_idx} > 数据长度={data.shape[-1]}，将截断")
            end_idx = data.shape[-1]
        sliced = data[..., start_idx:end_idx]
        self.logger.debug(f"Slicing: data{data.shape} -> sliced{sliced.shape}, window_time={window_time}s")
        return sliced

class SSVEPModelManager:
    def __init__(self, config: SSVEPConfig, model_path: str = "../temp"):
        self.config = config
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    def create_model(self) -> torch.nn.Module:
        model = TENet(num_classes=self.config.Nf, num_channels=self.config.Nc,filters=self.config.filters)
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
        # Fit TDCA on raw cross-subject data
        estimator1 = TDCA(padding_len=5, n_components=8)
        estimator1.fit(X_train, y_train, Yf)
        _, X_cross_train = estimator1.transform(X_train)
        _, X_intra_test = estimator1.transform(test_data)
        X_cross_train = np.squeeze(X_cross_train, axis=2)
        X_intra_test = np.squeeze(X_intra_test, axis=2)
        model = TENet(num_classes=self.config.Nf, num_channels=self.config.Nc)
        model = model.to(self.device)
        train_loader = self.create_dataloader(X_cross_train, y_train, self.config.bz1, True)
        test_loader = self.create_dataloader(X_intra_test, test_labels, 20, False)
        val_acc, kappa, recall, precision = self.train_stage(
            model, train_loader, test_loader, self.config.epochs1, self.config.lr
        )

        self.logger.info("Stage 2: Intra-Subject Fine-tuning")

        # Fit NEW TDCA on raw intra-subject training data
        estimator2 = TDCA(padding_len=5, n_components=8)
        estimator2.fit(train_data, train_labels, Yf)
        _, X_intra_train = estimator2.transform(train_data)
        _, X_intra_test2 = estimator2.transform(test_data)
        X_intra_train = np.squeeze(X_intra_train, axis=2)
        X_intra_test2 = np.squeeze(X_intra_test2, axis=2)

        # 替换全连接层 - 确保使用正确的输入维度
        model.eval()
        with torch.no_grad():
            test_input = torch.tensor(X_intra_train[:1], dtype=torch.float32).to(self.device)
            if test_input.dim() == 3:
                test_input = test_input.unsqueeze(1)  # 添加 Nm 维度
            features = model.chan_feature_extraction(test_input)
            features = model.time1_feature_extraction(features)
            features = model.time2_feature_extraction(features)
            actual_feature_dim = features.view(features.size(0), -1).shape[1]

        model.dense_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            LinearWithConstraint(actual_feature_dim, self.config.Nf, max_norm=0.5)
        ).to(self.device)
        model = model.to(self.device)
        train_loader = self.create_dataloader(X_intra_train, train_labels, self.config.bz2, True)
        test_loader = self.create_dataloader(X_intra_test2, test_labels, 20, False)

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
            freqs=self.config.freq_list, srate=self.config.Fs, T=window_time, n_harmonics=5
        )
    def evaluate_performance(self, acc: float, window_time: float) -> float:
        from Utils.itr import itr
        return itr(self.config.Nf, float(acc), 0.5 + window_time)

    def save_results(self, results: dict, window_time: float, dataset_name: str):
        for metric, values in results.items():
            df = pd.DataFrame(values)
            save_path = self.output_dir / f"{dataset_name}_{window_time:.1f}s_{metric}.csv"
            df.to_csv(save_path, index=False)
            self.logger.info(f"Saved {metric} to {save_path}")

    def run(self):
        #train_blocks和test_blocks可以手动改
        train_block=4
        test_block=2
        #上面两个数据集的数据划分参数需要动修改
        self.logger.info(f"Starting SSVEP Experiment with dataset: {self.config.dataset}")
        if self.config.window_times is None or self.config.data_lengths is None:
            raise ValueError("window_times和data_lengths未正确初始化")
        for idx, window_time in enumerate(self.config.window_times):
            self.logger.info(f"\n=== Window Time: {window_time}s (data_length: {self.config.data_lengths[idx]}) ===")
            Yf = self.generate_reference_signals(window_time)
            results = {metric: [] for metric in ['acc', 'itr', 'kappa', 'precision', 'recall']}
            for test_subject in range(1, self.config.Ns + 1):
                self.logger.info(f"Test Subject: {test_subject}")
                try:
                    X_train, y_train = self.data_manager.prepare_cross_subject_data(
                        test_subject, window_time,
                    )
                    if self.config.dataset=="Benchmark":
                        train_data, train_labels, test_data, test_labels = \
                            self.data_manager.prepare_intra_subject_data(
                                test_subject=test_subject, train_block=train_block, test_block=test_block,
                                window_time=window_time,
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
                except Exception as e:
                    self.logger.error(f"Error processing subject {test_subject}: {e}")
                    # 添加默认值以避免中断
                    results['acc'].append(0.0)
                    results['itr'].append(0.0)
                    results['kappa'].append(0.0)
                    results['precision'].append(0.0)
                    results['recall'].append(0.0)

            self.save_results(results, window_time, self.config.dataset)
            self.logger.info(
                f"=== Summary for {window_time}s ===\n"
                f"Average ACC: {np.mean(results['acc']):.4f} | "
                f"Average ITR: {np.mean(results['itr']):.2f} | "
                f"Average Kappa: {np.mean(results['kappa']):.4f}"
            )

# ==================== 主程序 ====================
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    # 配置Benchmark数据集
    benchmark_config = SSVEPConfig(
        dataset='Benchmark',
        Fs=250, Nc=8, Nf=40, Ns=35, epochs1=500, epochs2=300,
        bz1=40, bz2=20, lr=0.001, trans_lr=0.0001,
        data_root=str(current_dir / "C:/Users/WQER/Desktop/ssvep_data40"),
        window_times=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    config = benchmark_config  # 或 new_dataset_config
    print(f"当前工作目录: {current_dir.absolute()}")
    print(f"数据根目录: {config.data_root}")
    print(f"数据集: {config.dataset}")
    print(f"窗口时间: {config.window_times}")
    print(f"自动计算长度: {config.data_lengths}")
    experiment = SSVEPExperiment(config, output_dir=f"../results_{config.dataset}")
    experiment.run()