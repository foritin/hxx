# 🚀 Python项目配置指南

## 📋 项目概述

这是一个使用Poetry管理依赖的Python项目，配置了自动格式化工具（Black + isort）和VS Code开发环境。

## 🛠️ 开发环境设置

### 1. 安装依赖

```bash
# 安装Poetry（如果还没有）
curl -sSL https://install.python-poetry.org | python3 -

# 安装项目依赖
poetry install

# 激活虚拟环境
poetry shell
```

### 2. VS Code配置

项目已经配置了`.vscode/settings.json`，包含：
- ✅ Python解释器自动检测
- ✅ 保存时自动格式化（Black + isort）
- ✅ 自动组织导入语句
- ✅ 代码检查（flake8 + mypy）
- ✅ 项目路径配置

### 3. 代码格式化

#### 自动格式化（推荐）
- 在VS Code中编辑Python文件时，保存文件会自动：
  - 使用Black格式化代码
  - 使用isort整理导入语句

#### 手动格式化
```bash
# 格式化所有代码
poetry run black .

# 整理导入语句
poetry run isort .

# 代码检查
poetry run flake8 .
poetry run mypy .
```

## 🗂️ 项目结构

```
hxx/
├── engine/          # 核心引擎模块
│   └── __init__.py
├── module/          # 业务逻辑模块
│   └── __init__.py
├── main.py          # 主程序入口
├── pyproject.toml   # Poetry配置和工具设置
├── .vscode/         # VS Code配置
│   └── settings.json
└── README.md        # 项目说明
```

## 🎯 运行程序

```bash
# 在项目根目录执行
python main.py

# 或者使用poetry
poetry run python main.py
```

## 📦 依赖管理

### 添加新依赖

```bash
# 添加生产依赖
poetry add requests numpy pandas

# 添加开发依赖
poetry add --group dev pytest-xdist sphinx

# 添加测试依赖
poetry add --group test pytest-mock
```

### 更新依赖

```bash
# 更新所有依赖
poetry update

# 更新特定依赖
poetry update requests
```

## 🔧 工具配置

### Black配置
- 行长度: 88字符
- 目标Python版本: 3.13
- 排除目录: `.eggs`, `.git`, `.mypy_cache`, `.tox`, `.venv`, `build`, `dist`

### isort配置
- 兼容Black的配置
- 自动识别第一方包: `engine`, `module`
- 多行导入风格: 3

### MyPy配置
- 严格类型检查
- 警告未使用的配置
- 不允许未类型化的函数定义

## 📝 开发建议

1. **编码规范**: 项目使用Black和isort自动格式化，请保持代码风格一致
2. **类型提示**: 推荐使用类型提示，MyPy会进行静态类型检查
3. **测试**: 在`tests/`目录添加测试文件，使用pytest运行测试
4. **文档**: 为函数和类添加文档字符串

## 🚀 快速开始

1. 克隆项目后运行 `poetry install`
2. 在VS Code中打开项目
3. 创建新的Python文件时会自动应用格式化配置
4. 编辑代码时保存文件会自动格式化

## 💡 提示

- 如果VS Code没有自动检测到Python解释器，请手动选择`.venv/bin/python`
- 格式化工具的配置在`pyproject.toml`中，可以根据需要调整
- 建议安装VS Code的Python扩展和Black Formatter扩展
