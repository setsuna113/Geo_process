# 核心功能依赖问题分析

## 问题汇总

### 1. scripts/run_analysis.py 的4个关键依赖
```python
Line 39: from src.processors.data_preparation.standalone_analysis_datasets import create_analysis_dataset
Line 41: from src.core.enhanced_progress_manager import EnhancedProgressManager  
Line 42: from src.pipelines.stages.analyzer_factory import AnalyzerFactory
Line 43: from src.infrastructure.monitoring.unified_monitor import UnifiedMonitor
```

### 2. src/gee_integration/config.py 的日志依赖
```python
Line 13: from src.infrastructure.logging import get_logger
```

### 3. src/base/__init__.py 的文档引用
```python
Line 30: from src.core import component_registry  # 示例代码中引用
```

## 依赖分析

### run_analysis.py 是什么？
- **主要分析入口脚本**
- 用于运行 SOM、GWPCA、MaxP 等空间分析
- 设计为"standalone"但仍依赖pipeline组件

### 这些依赖的作用：

1. **standalone_analysis_datasets** - 创建分析数据集
   - 处理parquet文件
   - 准备数据用于分析

2. **EnhancedProgressManager** - 进度管理
   - 跟踪分析进度
   - 可以无数据库运行（use_database=False）

3. **AnalyzerFactory** - 分析器工厂
   - 创建SOM、GWPCA等分析器实例
   - 核心分析功能的入口

4. **UnifiedMonitor** - 监控系统
   - 资源监控
   - 性能跟踪

## 解决方案对比

### 方案A：恢复必要文件（推荐）

**恢复这些特定文件**：
```bash
# 恢复4个关键文件
git restore src/processors/data_preparation/standalone_analysis_datasets.py
git restore src/core/enhanced_progress_manager.py  
git restore src/pipelines/stages/analyzer_factory.py
git restore src/infrastructure/monitoring/unified_monitor.py

# 恢复日志模块
git restore src/infrastructure/logging/

# 检查这些文件的依赖并递归恢复
```

**优点**：
- ✅ 最小化恢复，保持大部分清理
- ✅ 立即修复功能
- ✅ 不需要重写代码

**缺点**：
- ❌ 可能需要恢复更多依赖文件
- ❌ 不够"干净"

### 方案B：迁移到简化版本

**创建简化的替代实现**：

```python
# src/analysis/data_loader.py - 替代 standalone_analysis_datasets
class SimpleDataLoader:
    @staticmethod
    def load_parquet(path):
        import pandas as pd
        return pd.read_parquet(path)

# src/analysis/progress.py - 替代 EnhancedProgressManager
class SimpleProgressTracker:
    def __init__(self):
        self.current = 0
        self.total = 100
    
    def update(self, n=1):
        self.current += n
        print(f"Progress: {self.current}/{self.total}")

# src/analysis/factory.py - 替代 AnalyzerFactory
class SimpleAnalyzerFactory:
    @staticmethod
    def create(method_name):
        if method_name == "som":
            from src.biodiversity_analysis.methods.som import SOMAnalyzer
            return SOMAnalyzer()
        # ... etc

# src/analysis/monitor.py - 替代 UnifiedMonitor
class SimpleMonitor:
    def log_memory(self):
        import psutil
        print(f"Memory: {psutil.virtual_memory().percent}%")
```

**优点**：
- ✅ 真正的清理，无冗余代码
- ✅ 简单清晰的实现
- ✅ 无复杂依赖

**缺点**：
- ❌ 需要重写大量代码
- ❌ 可能丢失功能
- ❌ 需要大量测试

### 方案C：混合方案（最实用）

1. **恢复最少必要文件**
2. **简化其依赖**
3. **创建迁移脚本**

```bash
# 1. 恢复核心文件
git restore src/processors/data_preparation/standalone_analysis_datasets.py
git restore src/pipelines/stages/analyzer_factory.py

# 2. 创建简化替代
# - 用简单日志替代 infrastructure.logging
# - 用简单进度条替代 EnhancedProgressManager
# - 用简单监控替代 UnifiedMonitor

# 3. 修改 run_analysis.py 使用简化版本
```

## Linus式判断

**"Perfect is the enemy of good."**

### 推荐：方案C（混合方案）

理由：
1. **实用主义** - 快速修复，后续改进
2. **最小恢复** - 只恢复真正需要的
3. **逐步简化** - 不是一次性重写

### 具体步骤：

```bash
# Step 1: 恢复2个核心文件
git restore src/processors/data_preparation/standalone_analysis_datasets.py
git restore src/pipelines/stages/analyzer_factory.py

# Step 2: 检查它们的依赖
# 看看还需要恢复什么

# Step 3: 创建简化的替代品
# - simple_logger.py 替代 infrastructure.logging
# - simple_progress.py 替代 enhanced_progress_manager
# - simple_monitor.py 替代 unified_monitor
```

## 决定

需要决定：
1. ✅ 恢复最少文件（方案C）
2. ❌ 全部重写（太耗时）
3. ❌ 恢复整个模块（违背清理初衷）

**核心原则**："不要破坏可用的功能"