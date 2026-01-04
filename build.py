#!/usr/bin/env python3
"""
Kiro Proxy 跨平台打包脚本
支持: Windows / macOS / Linux (通用)

使用方法:
    python build.py          # 打包当前平台
    python build.py --all    # 显示所有平台打包说明
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

APP_NAME = "KiroProxy"
VERSION = "2.0.0"
MAIN_SCRIPT = "app.py"
ICON_DIR = Path("assets")

def get_platform():
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    else:
        return "linux"

def ensure_pyinstaller():
    """确保 PyInstaller 已安装"""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} 已安装")
    except ImportError:
        print("→ 安装 PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

def clean_build():
    """清理构建目录"""
    for d in ["build", "dist", f"{APP_NAME}.spec"]:
        if os.path.isdir(d):
            shutil.rmtree(d)
        elif os.path.isfile(d):
            os.remove(d)
    print("✓ 清理完成")

def build_app():
    """构建应用"""
    platform = get_platform()
    print(f"\n{'='*50}")
    print(f"  构建 {APP_NAME} v{VERSION} - {platform}")
    print(f"{'='*50}\n")
    
    ensure_pyinstaller()
    clean_build()
    
    # PyInstaller 参数
    args = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onefile",           # 单文件
        "--clean",             # 清理缓存
        "--noconfirm",         # 不确认覆盖
        # 隐藏控制台（Windows）
        # "--noconsole" if platform == "windows" else "",
    ]
    
    # 添加图标（如果存在）
    icon_file = None
    if platform == "windows" and (ICON_DIR / "icon.ico").exists():
        icon_file = ICON_DIR / "icon.ico"
    elif platform == "macos" and (ICON_DIR / "icon.icns").exists():
        icon_file = ICON_DIR / "icon.icns"
    elif (ICON_DIR / "icon.png").exists():
        icon_file = ICON_DIR / "icon.png"
    
    if icon_file:
        args.extend(["--icon", str(icon_file)])
        print(f"✓ 使用图标: {icon_file}")
    
    # 添加隐藏导入（确保所有依赖被打包）
    hidden_imports = [
        "uvicorn.logging",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "httpx",
        "httpx._transports",
        "httpx._transports.default",
        "anyio",
        "anyio._backends",
        "anyio._backends._asyncio",
    ]
    for imp in hidden_imports:
        args.extend(["--hidden-import", imp])
    
    # 主脚本
    args.append(MAIN_SCRIPT)
    
    # 过滤空参数
    args = [a for a in args if a]
    
    print(f"→ 执行: {' '.join(args)}\n")
    result = subprocess.run(args)
    
    if result.returncode == 0:
        # 输出文件
        if platform == "windows":
            output = Path("dist") / f"{APP_NAME}.exe"
        else:
            output = Path("dist") / APP_NAME
        
        if output.exists():
            size_mb = output.stat().st_size / (1024 * 1024)
            print(f"\n{'='*50}")
            print(f"  ✓ 构建成功!")
            print(f"  输出: {output}")
            print(f"  大小: {size_mb:.1f} MB")
            print(f"{'='*50}")
            
            # 创建发布包
            create_release_package(platform, output)
        else:
            print("✗ 构建失败: 输出文件不存在")
            sys.exit(1)
    else:
        print("✗ 构建失败")
        sys.exit(1)

def create_release_package(platform, binary_path):
    """创建发布包"""
    release_dir = Path("release")
    release_dir.mkdir(exist_ok=True)
    
    if platform == "windows":
        # Windows: zip 包
        archive_name = f"{APP_NAME}-{VERSION}-Windows"
        shutil.copy(binary_path, release_dir / f"{APP_NAME}.exe")
        shutil.make_archive(
            str(release_dir / archive_name),
            "zip",
            release_dir,
            f"{APP_NAME}.exe"
        )
        (release_dir / f"{APP_NAME}.exe").unlink()
        print(f"  发布包: release/{archive_name}.zip")
        
    elif platform == "macos":
        # macOS: zip 包
        archive_name = f"{APP_NAME}-{VERSION}-macOS"
        shutil.copy(binary_path, release_dir / APP_NAME)
        os.chmod(release_dir / APP_NAME, 0o755)
        shutil.make_archive(
            str(release_dir / archive_name),
            "zip",
            release_dir,
            APP_NAME
        )
        (release_dir / APP_NAME).unlink()
        print(f"  发布包: release/{archive_name}.zip")
        
    else:
        # Linux: tar.gz 包
        archive_name = f"{APP_NAME}-{VERSION}-Linux"
        shutil.copy(binary_path, release_dir / APP_NAME)
        os.chmod(release_dir / APP_NAME, 0o755)
        shutil.make_archive(
            str(release_dir / archive_name),
            "gztar",
            release_dir,
            APP_NAME
        )
        (release_dir / APP_NAME).unlink()
        print(f"  发布包: release/{archive_name}.tar.gz")

def show_all_platforms():
    """显示所有平台打包说明"""
    print(f"""
{'='*60}
  Kiro Proxy 跨平台打包说明
{'='*60}

本脚本需要在目标平台上运行才能打包对应平台的版本。

【Windows 打包】
  在 Windows 上运行:
    python build.py
  
  输出: release/KiroProxy-{VERSION}-Windows.zip

【macOS 打包】
  在 macOS 上运行:
    python build.py
  
  输出: release/KiroProxy-{VERSION}-macOS.zip

【Linux 打包】
  在 Linux 上运行:
    python build.py
  
  输出: release/KiroProxy-{VERSION}-Linux.tar.gz
  
  注: Linux 版本通用，适用于大多数发行版
      (Ubuntu, Debian, Fedora, Arch, CentOS 等)

【GitHub Actions 自动构建】
  推送到 GitHub 后，Actions 会自动构建所有平台版本。
  见 .github/workflows/build.yml

{'='*60}
""")

if __name__ == "__main__":
    if "--all" in sys.argv or "-a" in sys.argv:
        show_all_platforms()
    else:
        build_app()
