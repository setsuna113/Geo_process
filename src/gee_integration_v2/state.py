"""
状态管理模块
===========

处理流程状态的保存和恢复，支持断点续传。

作者: Linus式重构
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """流程状态 - 用于恢复"""
    step: str  # 'upload', 'process', 'export', 'download', 'complete'
    asset_ids: Dict[str, str] = field(default_factory=dict)
    processed_images: Dict[str, str] = field(default_factory=dict)  # resolution -> image_id
    export_tasks: Dict[str, str] = field(default_factory=dict)  # resolution -> task_id
    completed_exports: Dict[str, Path] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime('%Y%m%d_%H%M%S'))
    
    def save(self, path: Path):
        """保存状态到JSON"""
        state_dict = {
            'step': self.step,
            'asset_ids': self.asset_ids,
            'processed_images': self.processed_images,
            'export_tasks': self.export_tasks,
            'completed_exports': {k: str(v) for k, v in self.completed_exports.items()},
            'timestamp': self.timestamp
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2)
        logger.info(f"Saved state to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'PipelineState':
        """从JSON加载状态"""
        with open(path) as f:
            state_dict = json.load(f)
        
        # 转换路径
        if 'completed_exports' in state_dict:
            state_dict['completed_exports'] = {
                k: Path(v) for k, v in state_dict['completed_exports'].items()
            }
        
        return cls(**state_dict)
    
    @classmethod
    def get_latest(cls, state_dir: Path) -> Optional['PipelineState']:
        """获取最新的状态文件"""
        state_files = list(state_dir.glob('state_*.json'))
        if not state_files:
            return None
        
        latest = max(state_files, key=lambda p: p.stat().st_mtime)
        return cls.load(latest)
    
    def can_resume_from(self, target_step: str) -> bool:
        """检查是否可以从目标步骤恢复"""
        steps_order = ['upload', 'process', 'export', 'download', 'complete']
        
        if self.step not in steps_order or target_step not in steps_order:
            return False
        
        current_idx = steps_order.index(self.step)
        target_idx = steps_order.index(target_step)
        
        return current_idx >= target_idx